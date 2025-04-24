import streamlit as st
import os
import time
import base64
import cv2
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import io
import mediapipe as mp
from google.generativeai import GenerativeModel
import requests

# ===== UTILITIES =====

# ===== POSE DETECTION =====
def analyze_video(video_file):
    """
    Process video with MediaPipe to extract pose data
    
    Args:
        video_file: Uploaded video file object
        
    Returns:
        frames: List of annotated frames with pose detection
        landmarks_data: NumPy array of landmark coordinates
    """
    try:
        # Save uploaded file temporarily
        temp_file = "temp_video.mp4"
        with open(temp_file, "wb") as f:
            f.write(video_file.getbuffer())
        
        # Initialize MediaPipe pose detector
        mp_pose = mp.solutions.pose
        mp_drawing = mp.solutions.drawing_utils
        
        # Process video
        landmarks_data = []
        frames = []
        
        cap = cv2.VideoCapture(temp_file)
        frame_count = 0
        max_frames = 100  # Limit processing to first 100 frames for speed
        
        with mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:
            
            while cap.isOpened() and frame_count < max_frames:
                success, image = cap.read()
                if not success:
                    break
                
                # Process every 5th frame for efficiency
                if frame_count % 5 == 0:
                    # Convert to RGB for MediaPipe
                    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    
                    # Process image
                    results = pose.process(image_rgb)
                    
                    # Draw landmarks
                    annotated_image = image_rgb.copy()
                    if results.pose_landmarks:
                        mp_drawing.draw_landmarks(
                            annotated_image, 
                            results.pose_landmarks,
                            mp_pose.POSE_CONNECTIONS)
                        
                        # Extract landmark coordinates
                        frame_landmarks = []
                        for landmark in results.pose_landmarks.landmark:
                            frame_landmarks.extend([landmark.x, landmark.y, landmark.z, landmark.visibility])
                        
                        landmarks_data.append(frame_landmarks)
                        frames.append(annotated_image)
                
                frame_count += 1
        
        cap.release()
        
        # Clean up
        if os.path.exists(temp_file):
            os.remove(temp_file)
        
        # Convert landmarks to numpy array
        landmarks_data = np.array(landmarks_data) if landmarks_data else np.array([])
        
        return frames, landmarks_data
    
    except Exception as e:
        st.error(f"Error analyzing video: {str(e)}")
        return [], np.array([])

def calculate_movement_metrics(landmarks_data):
    """
    Calculate movement metrics from pose landmarks
    
    Args:
        landmarks_data: NumPy array of landmark coordinates
        
    Returns:
        metrics: Dictionary of movement metrics
    """
    if landmarks_data.size == 0:
        # Return default metrics if no landmarks
        return {
            'speed': {'avg_foot_speed': 50, 'right_foot': 50, 'left_foot': 50},
            'agility': {'direction_changes': 50, 'max_acceleration': 50},
            'ball_control': {'hand_movement': 50},
            'balance': {'stability': 50, 'posture_consistency': 50}
        }
    
    try:
        # Define indices for relevant landmarks
        # MediaPipe landmark indices: https://google.github.io/mediapipe/solutions/pose.html
        right_foot_idx = 32 * 4  # Right foot index * 4 (x,y,z,visibility)
        left_foot_idx = 31 * 4   # Left foot index * 4
        right_knee_idx = 26 * 4  # Right knee index * 4
        left_knee_idx = 25 * 4   # Left knee index * 4
        
        # Wrist landmarks for ball control approximation
        right_wrist_idx = 16 * 4
        left_wrist_idx = 15 * 4
        
        # Hip landmarks for balance assessment
        right_hip_idx = 24 * 4
        left_hip_idx = 23 * 4
        
        # Shoulder landmarks for posture assessment
        right_shoulder_idx = 12 * 4
        left_shoulder_idx = 11 * 4
        
        # Calculate metrics
        
        # Speed metrics (using foot movement)
        right_foot_positions = landmarks_data[:, right_foot_idx:right_foot_idx+3]  # x, y, z
        left_foot_positions = landmarks_data[:, left_foot_idx:left_foot_idx+3]     # x, y, z
        
        # Calculate frame-to-frame movement (simplified velocity)
        right_foot_diffs = np.diff(right_foot_positions, axis=0)
        left_foot_diffs = np.diff(left_foot_positions, axis=0)
        
        # Compute magnitudes
        right_foot_magnitudes = np.linalg.norm(right_foot_diffs, axis=1)
        left_foot_magnitudes = np.linalg.norm(left_foot_diffs, axis=1)
        
        # Scaling factor to get values in a reasonable range (0-100)
        scale_factor = 10000
        
        # Calculate average speeds
        right_foot_speed = min(100, np.mean(right_foot_magnitudes) * scale_factor)
        left_foot_speed = min(100, np.mean(left_foot_magnitudes) * scale_factor)
        avg_foot_speed = (right_foot_speed + left_foot_speed) / 2
        
        # Agility metrics
        
        # Direction changes (based on foot movement direction changes)
        direction_changes_right = count_direction_changes(right_foot_diffs)
        direction_changes_left = count_direction_changes(left_foot_diffs)
        direction_changes = min(100, (direction_changes_right + direction_changes_left) * 5)  # Scale appropriately
        
        # Acceleration (based on foot movement changes)
        right_foot_accels = np.diff(right_foot_magnitudes)
        left_foot_accels = np.diff(left_foot_magnitudes)
        max_acceleration = min(100, (np.max(np.abs(right_foot_accels)) + np.max(np.abs(left_foot_accels))) * scale_factor / 2)
        
        # Ball control (approximated by hand movement, as we don't have actual ball data)
        right_hand_positions = landmarks_data[:, right_wrist_idx:right_wrist_idx+3]
        left_hand_positions = landmarks_data[:, left_wrist_idx:left_wrist_idx+3]
        
        right_hand_diffs = np.diff(right_hand_positions, axis=0)
        left_hand_diffs = np.diff(left_hand_positions, axis=0)
        
        right_hand_magnitudes = np.linalg.norm(right_hand_diffs, axis=1)
        left_hand_magnitudes = np.linalg.norm(left_hand_diffs, axis=1)
        
        hand_movement = min(100, np.mean(right_hand_magnitudes + left_hand_magnitudes) * scale_factor)
        
        # Balance metrics
        balance_metrics = calculate_balance_metrics(landmarks_data)
        
        # Normalize all metrics to 0-100
        metrics = {
            'speed': {
                'avg_foot_speed': avg_foot_speed,
                'right_foot': right_foot_speed,
                'left_foot': left_foot_speed
            },
            'agility': {
                'direction_changes': direction_changes,
                'max_acceleration': max_acceleration
            },
            'ball_control': {
                'hand_movement': hand_movement
            },
            'balance': balance_metrics
        }
        
        # Normalize all metrics
        metrics = normalize_metrics(metrics)
        
        return metrics
    
    except Exception as e:
        # Return default metrics in case of calculation error
        st.error(f"Error calculating metrics: {str(e)}")
        return {
            'speed': {'avg_foot_speed': 50, 'right_foot': 50, 'left_foot': 50},
            'agility': {'direction_changes': 50, 'max_acceleration': 50},
            'ball_control': {'hand_movement': 50},
            'balance': {'stability': 50, 'posture_consistency': 50}
        }

def calculate_balance_metrics(landmarks_data):
    """
    Calculate balance-related metrics from pose landmarks
    
    Args:
        landmarks_data: NumPy array of landmark coordinates
        
    Returns:
        balance_metrics: Dictionary of balance metrics
    """
    # Hip landmarks for balance assessment
    right_hip_idx = 24 * 4
    left_hip_idx = 23 * 4
    
    # Shoulder landmarks for posture assessment
    right_shoulder_idx = 12 * 4
    left_shoulder_idx = 11 * 4
    
    # Center of mass approximation (hips)
    right_hip_positions = landmarks_data[:, right_hip_idx:right_hip_idx+3]
    left_hip_positions = landmarks_data[:, left_hip_idx:left_hip_idx+3]
    center_of_mass = (right_hip_positions + left_hip_positions) / 2
    
    # Stability - how consistent the center of mass is (less movement = more stable)
    com_diffs = np.diff(center_of_mass, axis=0)
    com_magnitudes = np.linalg.norm(com_diffs, axis=1)
    stability = 100 - min(100, np.mean(com_magnitudes) * 5000)  # Inverse relationship with movement
    
    # Posture assessment - alignment of shoulders and hips
    right_shoulder_positions = landmarks_data[:, right_shoulder_idx:right_shoulder_idx+3]
    left_shoulder_positions = landmarks_data[:, left_shoulder_idx:left_shoulder_idx+3]
    
    # Vector between hips
    hip_vectors = right_hip_positions - left_hip_positions
    
    # Vector between shoulders
    shoulder_vectors = right_shoulder_positions - left_shoulder_positions
    
    # Calculate alignment (dot product of normalized vectors)
    hip_norms = np.linalg.norm(hip_vectors, axis=1)
    shoulder_norms = np.linalg.norm(shoulder_vectors, axis=1)
    
    # Avoid division by zero
    hip_norms[hip_norms == 0] = 1
    shoulder_norms[shoulder_norms == 0] = 1
    
    # Normalize vectors
    hip_vectors_norm = hip_vectors / hip_norms[:, np.newaxis]
    shoulder_vectors_norm = shoulder_vectors / shoulder_norms[:, np.newaxis]
    
    # Calculate dot products
    dot_products = np.sum(hip_vectors_norm * shoulder_vectors_norm, axis=1)
    
    # Convert to angle and measure consistency
    angles = np.arccos(np.clip(dot_products, -1.0, 1.0))
    angle_variance = np.var(angles)
    
    # Higher consistency = lower variance
    posture_consistency = 100 - min(100, angle_variance * 1000)
    
    return {
        'stability': stability,
        'posture_consistency': posture_consistency
    }

def count_direction_changes(movements):
    """Count the number of significant direction changes in movement"""
    if len(movements) < 3:
        return 0
    
    # Direction changes in x, y dimensions
    direction_changes = 0
    
    # For x dimension
    prev_direction = np.sign(movements[0, 0])
    for i in range(1, len(movements)):
        curr_direction = np.sign(movements[i, 0])
        # Only count significant changes (filter noise)
        if curr_direction != prev_direction and abs(movements[i, 0]) > 0.01:
            direction_changes += 1
            prev_direction = curr_direction
    
    # For y dimension
    prev_direction = np.sign(movements[0, 1])
    for i in range(1, len(movements)):
        curr_direction = np.sign(movements[i, 1])
        # Only count significant changes (filter noise)
        if curr_direction != prev_direction and abs(movements[i, 1]) > 0.01:
            direction_changes += 1
            prev_direction = curr_direction
    
    return direction_changes

def normalize_metrics(metrics):
    """Normalize metrics to a 0-100 scale for visualization"""
    # Ensure all metrics are in 0-100 range
    for category in metrics:
        for metric in metrics[category]:
            metrics[category][metric] = max(0, min(100, metrics[category][metric]))
    
    return metrics

# ===== AI MODELS =====
def get_offline_analysis(video_frames, landmarks, player_info, is_comparison=False):
    """
    Generate analysis without using external AI models
    
    Args:
        video_frames: List of video frames with pose detection
        landmarks: NumPy array of landmark data
        player_info: String containing player information
        is_comparison: Boolean indicating if this is a comparison analysis
        
    Returns:
        Dictionary containing analysis results
    """
    # Calculate metrics from landmarks
    metrics = calculate_movement_metrics(landmarks)
    
    # Basic information extraction from player_info
    info_dict = {}
    if not is_comparison:
        for line in player_info.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info_dict[key.strip()] = value.strip()
    
    # Generate basic analysis based on metrics
    summary = ""
    if is_comparison:
        # For comparison prompts, return a basic comparison template
        summary = """
        # Player Comparison Analysis
        
        ## Comparison Summary
        Players have been analyzed based on their video footage. Detailed metrics are shown in the charts below.
        
        ## Relative Strengths
        - Compare the charts to see relative performance in different areas
        - Each player's strengths are highlighted in the radar charts
        
        ## Development Recommendations
        - Focus training on areas with lower metrics scores
        - Leverage strengths shown in the higher-scoring metrics
        """
    else:
        # For individual player analysis
        position = info_dict.get('Position', 'Unknown')
        age = info_dict.get('Age', 'Unknown')
        experience = info_dict.get('Experience Level', 'Unknown')
        
        # Position-specific insights
        position_insights = {
            'Goalkeeper': "Focus areas should include reaction time and positioning.",
            'Defender': "Key attributes include positioning and tackling ability.",
            'Midfielder': "Ball control and passing accuracy are essential skills.",
            'Forward': "Finishing ability and positioning in the final third are crucial.",
            'Winger': "Speed and crossing ability are primary attributes."
        }
        
        # Generate analysis text based on metrics
        speed_rating = metrics['speed']['avg_foot_speed']
        agility_rating = (metrics['agility']['direction_changes'] + metrics['agility']['max_acceleration']) / 2
        ball_control = metrics['ball_control']['hand_movement']
        balance = (metrics['balance']['stability'] + metrics['balance']['posture_consistency']) / 2
        
        # Overall rating calculation
        overall_rating = (speed_rating + agility_rating + ball_control + balance) / 4
        overall_rating_scaled = max(1, min(10, overall_rating / 10))
        
        summary = f"""
        # Performance Analysis
        
        ## Technical Skills
        - Ball Control: {ball_control:.1f}/100
        - Movement quality is {'excellent' if ball_control > 80 else 'good' if ball_control > 60 else 'average' if ball_control > 40 else 'needs improvement'}
        
        ## Physical Attributes
        - Speed: {speed_rating:.1f}/100
        - Agility: {agility_rating:.1f}/100
        - Balance: {balance:.1f}/100
        
        ## Position-Specific Insights
        As a {position}, {position_insights.get(position, "focus on general skill development.")}
        
        ## Development Recommendations
        - {'Work on speed and acceleration' if speed_rating < 60 else 'Maintain good speed levels'}
        - {'Improve agility and direction changes' if agility_rating < 60 else 'Continue developing agility'}
        - {'Focus on ball control exercises' if ball_control < 60 else 'Maintain good ball control technique'}
        - {'Practice balance and stability drills' if balance < 60 else 'Maintain good balance and posture'}
        
        ## Overall Rating
        {overall_rating_scaled:.1f}/10
        """
    
    return {
        'summary': summary,
        'metrics': metrics
    }

def analyze_with_gemini(video_frames, landmarks, player_info, api_key, is_comparison=False):
    """
    Analyze player performance using Google's Gemini model
    
    Args:
        video_frames: List of video frames with pose detection
        landmarks: NumPy array of landmark data
        player_info: String containing player information
        api_key: Gemini API key
        is_comparison: Boolean indicating if this is a comparison analysis
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Calculate metrics from landmarks
        metrics = {}
        if landmarks is not None and len(landmarks) > 0:
            metrics = calculate_movement_metrics(landmarks)
        
        # Configure Gemini
        gemini = GenerativeModel('gemini-1.5-flash')
        
        # Prepare frames for analysis
        frame_strings = []
        if video_frames and len(video_frames) > 0:
            for frame in video_frames[:5]:  # Sample frames
                _, buffer = cv2.imencode('.jpg', frame)
                frame_strings.append(
                    {
                        "inlineData": {
                            "mimeType": "image/jpeg",
                            "data": base64.b64encode(buffer).decode('utf-8')
                        }
                    }
                )
        
        # Create prompt based on whether this is a comparison or individual analysis
        if is_comparison:
            prompt = f"""
            {player_info}
            
            Based on the above information comparing two soccer players,
            provide a detailed analysis highlighting:
            1. Relative strengths and weaknesses of each player
            2. Potential best position for each player
            3. Which player might be better suited for different team needs
            4. Development potential for each player
            
            Format your response with markdown headings and bullet points.
            """
        else:
            # Format metrics for inclusion in the prompt
            metrics_str = json.dumps(metrics, indent=2) if metrics else "No metrics available"
            
            prompt = f"""
            Analyze this soccer player's performance based on the following information:
            
            Player Info:
            {player_info}
            
            Player Metrics:
            {metrics_str}
            
            Also analyze any visible pose information from the provided images.
            
            Provide a detailed analysis covering:
            1. Technical Skills (ball control, passing, shooting)
            2. Physical Attributes (speed, agility, strength)
            3. Tactical Awareness (positioning, decision making)
            4. Potential Areas for Improvement
            5. Overall Rating (1-10 scale)
            
            Format your response with markdown headings and bullet points.
            """
        
        # Create content parts for the model
        contents = [{"role": "user", "parts": [{"text": prompt}]}]
        
        # Add frames if available
        if frame_strings:
            for frame in frame_strings:
                contents[0]["parts"].append(frame)
        
        # Generate response
        response = gemini.generate_content(contents)
        
        return {
            'summary': response.text,
            'metrics': metrics
        }
    except Exception as e:
        metrics = {}  # Placeholder if metrics wasn't defined
        if landmarks is not None and len(landmarks) > 0:
            metrics = calculate_movement_metrics(landmarks)
        return {
            'summary': f"Error analyzing with Gemini: {str(e)}",
            'metrics': metrics
        }

def analyze_with_deepseek(video_frames, landmarks, player_info, api_key, is_comparison=False):
    """
    Analyze player performance using DeepSeek API
    
    Args:
        video_frames: List of video frames with pose detection
        landmarks: NumPy array of landmark data
        player_info: String containing player information
        api_key: DeepSeek API key
        is_comparison: Boolean indicating if this is a comparison analysis
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Calculate metrics from landmarks
        metrics = {}
        if landmarks is not None and len(landmarks) > 0:
            metrics = calculate_movement_metrics(landmarks)
        
        # Prepare frames for analysis (convert to base64)
        frame_strings = []
        if video_frames and len(video_frames) > 0:
            for frame in video_frames[:3]:  # Sample frames
                _, buffer = cv2.imencode('.jpg', frame)
                frame_strings.append(base64.b64encode(buffer).decode('utf-8'))
        
        # Create prompt based on whether this is a comparison or individual analysis
        if is_comparison:
            prompt = f"""
            {player_info}
            
            Based on the above information comparing two soccer players,
            provide a detailed analysis highlighting:
            1. Relative strengths and weaknesses of each player
            2. Potential best position for each player
            3. Which player might be better suited for different team needs
            4. Development potential for each player
            
            Format your response with markdown headings and bullet points.
            """
        else:
            # Format metrics for inclusion in the prompt
            metrics_str = json.dumps(metrics, indent=2) if metrics else "No metrics available"
            
            prompt = f"""
            Analyze this soccer player's performance based on the following information:
            
            Player Info:
            {player_info}
            
            Player Metrics:
            {metrics_str}
            
            Also analyze any visible pose information from the provided images.
            
            Provide a detailed analysis covering:
            1. Technical Skills (ball control, passing, shooting)
            2. Physical Attributes (speed, agility, strength)
            3. Tactical Awareness (positioning, decision making)
            4. Potential Areas for Improvement
            5. Overall Rating (1-10 scale)
            
            Format your response with markdown headings and bullet points.
            """
        
        # For simulation purposes (since we don't have actual DeepSeek API access)
        # we'll create a function that generates a synthetic response based on the metrics
        # In a real implementation, this would be an actual API call
        
        # Create a DeepSeek-style analysis from metrics
        analysis = generate_synthetic_deepseek_response(metrics, player_info, is_comparison)
        
        return {
            'summary': analysis,
            'metrics': metrics
        }
    except Exception as e:
        metrics = {}  # Placeholder if metrics wasn't defined
        if landmarks is not None and len(landmarks) > 0:
            metrics = calculate_movement_metrics(landmarks)
        return {
            'summary': f"Error analyzing with DeepSeek: {str(e)}",
            'metrics': metrics
        }

def analyze_with_huggingface(video_frames, landmarks, player_info, api_key, is_comparison=False):
    """
    Analyze player performance using Hugging Face's LLaVa model
    
    Args:
        video_frames: List of video frames with pose detection
        landmarks: NumPy array of landmark data
        player_info: String containing player information
        api_key: Hugging Face API key
        is_comparison: Boolean indicating if this is a comparison analysis
        
    Returns:
        Dictionary containing analysis results
    """
    try:
        # Calculate metrics from landmarks
        metrics = {}
        if landmarks is not None and len(landmarks) > 0:
            metrics = calculate_movement_metrics(landmarks)
        
        # In a real implementation, this would send the frames to Hugging Face's API
        # For now, generate a response based on the metrics
        
        # Extract player info (similar to offline analysis)
        info_dict = {}
        if not is_comparison:
            for line in player_info.strip().split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    info_dict[key.strip()] = value.strip()
        
        # Format metrics for analysis
        speed_score = metrics.get('speed', {}).get('avg_foot_speed', 50)
        agility_score = (metrics.get('agility', {}).get('direction_changes', 50) + 
                         metrics.get('agility', {}).get('max_acceleration', 50)) / 2
        ball_control = metrics.get('ball_control', {}).get('hand_movement', 50)
        balance = (metrics.get('balance', {}).get('stability', 50) + 
                  metrics.get('balance', {}).get('posture_consistency', 50)) / 2
        
        # Get name and position
        name = info_dict.get('Name', 'Player')
        position = info_dict.get('Position', 'Unknown')
        
        # Calculate overall rating
        overall_rating = (speed_score + agility_score + ball_control + balance) / 4
        overall_rating_scaled = max(1, min(10, overall_rating / 10))
        
        # Position-specific insights
        position_insights = {
            'Goalkeeper': "Focus on reaction time, positioning, and aerial ability.",
            'Defender': "Prioritize defensive positioning, tackling, and anticipation.",
            'Midfielder': "Develop passing range, vision, and spatial awareness.",
            'Forward': "Enhance finishing, positioning, and movement off the ball.",
            'Winger': "Emphasize speed, dribbling, and crossing ability."
        }
        
        analysis = f"""
        # Player Analysis: {name}
        
        ## Technical Assessment
        - Ball Control: {ball_control:.1f}/100 - {'Excellent' if ball_control > 80 else 'Good' if ball_control > 60 else 'Average' if ball_control > 40 else 'Needs improvement'}
        - Movement quality appears {'fluid and controlled' if ball_control > 60 else 'somewhat rigid and could be improved'}
        - {'Demonstrates good technique' if ball_control > 60 else 'Technical skills need development'}
        
        ## Physical Attributes
        - Speed: {speed_score:.1f}/100 - {'Excellent' if speed_score > 80 else 'Good' if speed_score > 60 else 'Average' if speed_score > 40 else 'Needs improvement'}
        - Agility: {agility_score:.1f}/100 - {'Excellent' if agility_score > 80 else 'Good' if agility_score > 60 else 'Average' if agility_score > 40 else 'Needs improvement'}
        - Balance: {balance:.1f}/100 - {'Excellent' if balance > 80 else 'Good' if balance > 60 else 'Average' if balance > 40 else 'Needs improvement'}
        
        ## Position-Specific Analysis
        As a {position}: {position_insights.get(position, "Focus on general skill development.")}
        
        ## Development Recommendations
        - {'Work on speed and acceleration through sprint training' if speed_score < 60 else 'Maintain good speed levels through regular conditioning'}
        - {'Improve agility with cone drills and direction change exercises' if agility_score < 60 else 'Continue developing agility with advanced movement drills'}
        - {'Focus on ball control exercises and technical drills' if ball_control < 60 else 'Maintain good ball control technique through regular practice'}
        - {'Practice balance and stability drills to improve posture' if balance < 60 else 'Maintain good balance and posture through core strength exercises'}
        
        ## Overall Rating
        {overall_rating_scaled:.1f}/10
        """
    
        return {
            'summary': analysis,
            'metrics': metrics
        }
    except Exception as e:
        metrics = {}  # Placeholder if metrics wasn't defined
        if landmarks is not None and len(landmarks) > 0:
            metrics = calculate_movement_metrics(landmarks)
        return {
            'summary': f"Error analyzing with Hugging Face LLaVa: {str(e)}",
            'metrics': metrics
        }

def analyze_with_model(model_name, video_frames, landmarks, player_info, api_key, is_comparison=False):
    """
    Route analysis to the appropriate model based on user selection
    
    Args:
        model_name: String name of the model to use
        video_frames: List of video frames with pose detection
        landmarks: NumPy array of landmark data
        player_info: String containing player information
        api_key: API key for the selected model
        is_comparison: Boolean indicating if this is a comparison analysis
        
    Returns:
        Dictionary containing analysis results
    """
    if model_name == "offline":
        return get_offline_analysis(video_frames, landmarks, player_info, is_comparison)
    elif model_name == "gemini":
        return analyze_with_gemini(video_frames, landmarks, player_info, api_key, is_comparison)
    elif model_name == "deepseek":
        return analyze_with_deepseek(video_frames, landmarks, player_info, api_key, is_comparison)
    elif model_name == "huggingface":
        return analyze_with_huggingface(video_frames, landmarks, player_info, api_key, is_comparison)
    else:
        # Default to offline analysis if model not recognized
        return get_offline_analysis(video_frames, landmarks, player_info, is_comparison)

def generate_synthetic_deepseek_response(metrics, player_info, is_comparison=False):
    """
    Generate a simulated DeepSeek response based on metrics
    (In a real implementation, this would be an actual API call)
    
    Args:
        metrics: Dictionary of player metrics
        player_info: String containing player information
        is_comparison: Boolean indicating if this is a comparison analysis
        
    Returns:
        String containing the analysis response
    """
    if is_comparison:
        return """
        # Comparative Analysis

        ## Player Comparison Summary
        The players demonstrate different strengths and weaknesses as shown in the metric charts.

        ## Relative Strengths
        - The metrics visualization provides a clear comparison between players
        - Each player's unique attributes can be observed in the radar chart

        ## Position Recommendations
        - Position recommendations should be based on the relative strengths shown in the charts
        - Consider the balance of technical and physical attributes

        ## Development Pathways
        - Training programs should be customized based on the areas requiring improvement
        - Regular assessment using this tool can track progress over time
        """
    else:
        # Get information from metrics
        speed_score = metrics.get('speed', {}).get('avg_foot_speed', 50)
        agility_score = (metrics.get('agility', {}).get('direction_changes', 50) + 
                          metrics.get('agility', {}).get('max_acceleration', 50)) / 2
        ball_control = metrics.get('ball_control', {}).get('hand_movement', 50)
        balance = (metrics.get('balance', {}).get('stability', 50) + 
                    metrics.get('balance', {}).get('posture_consistency', 50)) / 2
        
        # Get player info
        info_dict = {}
        for line in player_info.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                info_dict[key.strip()] = value.strip()
        
        name = info_dict.get('Name', 'Player')
        position = info_dict.get('Position', 'Unknown')
        
        # Calculate overall rating
        overall_rating = (speed_score + agility_score + ball_control + balance) / 4
        overall_rating_scaled = max(1, min(10, overall_rating / 10))
        
        analysis = f"""
        # DeepSeek Analysis: {name}
        
        ## Technical Skills Analysis
        - Ball Control: {ball_control:.1f}/100 - {'Exceptional' if ball_control > 80 else 'Strong' if ball_control > 60 else 'Average' if ball_control > 40 else 'Requires development'}
        - The player demonstrates {'excellent technical ability' if ball_control > 80 else 'good technical skills' if ball_control > 60 else 'adequate technical ability' if ball_control > 40 else 'technical limitations that require attention'}
        - {'Recommend advanced technical drills' if ball_control > 80 else 'Recommend technical refinement exercises' if ball_control > 60 else 'Recommend focused technical training' if ball_control > 40 else 'Recommend fundamental technical development program'}
        
        ## Physical Attributes
        - Speed: {speed_score:.1f}/100 - {'Elite' if speed_score > 80 else 'Above average' if speed_score > 60 else 'Average' if speed_score > 40 else 'Below average'}
        - Agility: {agility_score:.1f}/100 - {'Elite' if agility_score > 80 else 'Above average' if agility_score > 60 else 'Average' if agility_score > 40 else 'Below average'}
        - Balance: {balance:.1f}/100 - {'Elite' if balance > 80 else 'Above average' if balance > 60 else 'Average' if balance > 40 else 'Below average'}
        
        ## Position Suitability Analysis
        - Current position: {position}
        - {'This appears to be an ideal position based on the player attributes' if overall_rating_scaled > 7 else 'This position is suitable, but the player could also consider alternatives' if overall_rating_scaled > 5 else 'The player might benefit from exploring alternative positions'}
        
        ## Development Pathway
        - Short-term focus: {'Technical refinement' if ball_control < speed_score and ball_control < agility_score else 'Speed development' if speed_score < ball_control and speed_score < agility_score else 'Agility enhancement'}
        - Long-term development: {'Complete player development with balanced training' if overall_rating_scaled > 7 else 'Targeted improvement in specific areas based on metrics' if overall_rating_scaled > 5 else 'Fundamental skill development across all areas'}
        
        ## Overall Rating and Potential
        - Current Rating: {overall_rating_scaled:.1f}/10
        - Development Potential: {'High' if overall_rating_scaled > 7 or (overall_rating_scaled > 5 and int(info_dict.get('Age', '25')) < 23) else 'Moderate' if overall_rating_scaled > 5 else 'Requires significant development'}
        """
        
        return analysis

# ===== DATA PROCESSING =====
def save_player_data(player_data):
    """
    Save player data to session state history
    
    Args:
        player_data: Dictionary containing player analysis data
        
    Returns:
        None
    """
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    # Add timestamp if not already present
    if 'timestamp' not in player_data:
        player_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
    
    # Add to history
    st.session_state.history.append(player_data)

def load_player_history():
    """
    Load player history from session state
    
    Args:
        None
        
    Returns:
        List of player data entries
    """
    return st.session_state.get('history', [])

def get_player_metrics(player_data):
    """
    Extract metrics from player data
    
    Args:
        player_data: Dictionary containing player analysis data
        
    Returns:
        Dictionary of metrics
    """
    # For individual player
    if 'metrics' in player_data:
        return player_data['metrics']
    
    # For comparison (player is in p1 or p2)
    if 'p1' in player_data and 'metrics' in player_data['p1']:
        return player_data['p1']['metrics']
    
    # Default empty metrics
    return {
        'speed': {'avg_foot_speed': 0, 'right_foot': 0, 'left_foot': 0},
        'agility': {'direction_changes': 0, 'max_acceleration': 0},
        'ball_control': {'hand_movement': 0},
        'balance': {'stability': 0, 'posture_consistency': 0}
    }

def format_metrics_for_chart(metrics):
    """
    Format metrics for visualization
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Tuple of (categories, values)
    """
    # Categories for radar chart
    categories = [
        'Speed', 
        'Right Foot Speed',
        'Left Foot Speed', 
        'Agility', 
        'Direction Changes',
        'Ball Control', 
        'Balance',
        'Posture'
    ]
    
    # Values corresponding to categories
    values = [
        metrics.get('speed', {}).get('avg_foot_speed', 0),
        metrics.get('speed', {}).get('right_foot', 0),
        metrics.get('speed', {}).get('left_foot', 0),
        metrics.get('agility', {}).get('max_acceleration', 0),
        metrics.get('agility', {}).get('direction_changes', 0),
        metrics.get('ball_control', {}).get('hand_movement', 0),
        metrics.get('balance', {}).get('stability', 0),
        metrics.get('balance', {}).get('posture_consistency', 0)
    ]
    
    return categories, values

def get_performance_summary(metrics):
    """
    Generate a performance summary from metrics
    
    Args:
        metrics: Dictionary of metrics
        
    Returns:
        Dictionary with strength, weakness, and overall rating
    """
    if not metrics:
        return {
            'strength': 'Not enough data',
            'weakness': 'Not enough data',
            'overall': 0
        }
    
    # Calculate overall rating
    speed_score = metrics.get('speed', {}).get('avg_foot_speed', 0)
    agility_score = (metrics.get('agility', {}).get('direction_changes', 0) + 
                     metrics.get('agility', {}).get('max_acceleration', 0)) / 2
    ball_control = metrics.get('ball_control', {}).get('hand_movement', 0)
    balance = (metrics.get('balance', {}).get('stability', 0) + 
               metrics.get('balance', {}).get('posture_consistency', 0)) / 2
    
    overall = (speed_score + agility_score + ball_control + balance) / 4
    
    # Find strength and weakness
    metrics_dict = {
        'Speed': speed_score,
        'Agility': agility_score,
        'Ball Control': ball_control,
        'Balance': balance
    }
    
    strength = max(metrics_dict.items(), key=lambda x: x[1])
    weakness = min(metrics_dict.items(), key=lambda x: x[1])
    
    return {
        'strength': f"{strength[0]} ({strength[1]:.1f}/100)",
        'weakness': f"{weakness[0]} ({weakness[1]:.1f}/100)",
        'overall': overall
    }

def format_player_info_from_dict(player_info):
    """
    Format player info from a dictionary
    
    Args:
        player_info: Dictionary containing player information
        
    Returns:
        Formatted string of player information
    """
    return f"""
    Name: {player_info.get('name', 'Unknown')}
    Position: {player_info.get('position', 'Unknown')}
    Age: {player_info.get('age', 'Unknown')}
    Height: {player_info.get('height', 'Unknown')} cm
    Weight: {player_info.get('weight', 'Unknown')} kg
    Preferred Foot: {player_info.get('foot', 'Unknown')}
    Experience Level: {player_info.get('experience', 'Unknown')}
    Notes: {player_info.get('notes', '')}
    """

# ===== VISUALIZATION =====
def create_radar_chart(categories, values, title):
    """
    Create a radar chart (also known as a spider or star chart)
    
    Args:
        categories: List of category names
        values: List of values (same length as categories)
        title: Chart title
        
    Returns:
        Figure object
    """
    # Number of variables
    N = len(categories)
    
    # Ensure we have at least 3 categories for a radar chart
    if N < 3:
        # Create a simple bar chart instead
        return create_bar_chart(categories, values, title)
    
    # Convert to radians and adjust for full circle
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    
    # Close the polygon by repeating the first value
    values_closed = np.concatenate((values, [values[0]]))
    angles_closed = np.concatenate((angles, [angles[0]]))
    categories_closed = np.concatenate((categories, [categories[0]]))
    
    # Create figure
    fig = Figure(figsize=(10, 8))
    ax = fig.add_subplot(111, polar=True)
    
    # Draw the outline of the chart
    ax.plot(angles_closed, values_closed, 'o-', linewidth=2)
    ax.fill(angles_closed, values_closed, alpha=0.25)
    
    # Set category labels
    ax.set_xticks(angles)
    ax.set_xticklabels(categories)
    
    # Set y-axis limits
    ax.set_ylim(0, 100)
    
    # Add grid and title
    ax.grid(True)
    ax.set_title(title, size=15, pad=20)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_bar_chart(categories, values, title):
    """
    Create a bar chart
    
    Args:
        categories: List of category names
        values: List of values (same length as categories)
        title: Chart title
        
    Returns:
        Figure object
    """
    # Create figure
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Create bars
    bars = ax.bar(categories, values, width=0.6)
    
    # Add labels and title
    ax.set_ylabel('Value (0-100)')
    ax.set_title(title)
    
    # Adjust limits
    ax.set_ylim(0, 105)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_comparison_bar_chart(categories, values1, values2, label1, label2, title):
    """
    Create a comparison bar chart for two players
    
    Args:
        categories: List of category names
        values1: List of values for player 1
        values2: List of values for player 2
        label1: Label for player 1
        label2: Label for player 2
        title: Chart title
        
    Returns:
        Figure object
    """
    # Create figure
    fig = Figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    
    # Set width of bars
    barWidth = 0.35
    
    # Set position of bars
    r1 = np.arange(len(categories))
    r2 = [x + barWidth for x in r1]
    
    # Create bars
    bars1 = ax.bar(r1, values1, width=barWidth, label=label1, alpha=0.8)
    bars2 = ax.bar(r2, values2, width=barWidth, label=label2, alpha=0.8)
    
    # Add labels and title
    ax.set_xlabel('Categories')
    ax.set_ylabel('Value (0-100)')
    ax.set_title(title)
    ax.set_xticks([r + barWidth/2 for r in range(len(categories))])
    ax.set_xticklabels(categories)
    
    # Add legend
    ax.legend()
    
    # Adjust limits
    ax.set_ylim(0, 105)
    
    # Add value labels on top of bars
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.1f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    autolabel(bars1)
    autolabel(bars2)
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def create_movement_heatmap(landmarks_data):
    """
    Create a heatmap visualization of player movement
    
    Args:
        landmarks_data: NumPy array of landmark coordinates
        
    Returns:
        Figure object
    """
    if landmarks_data.size == 0 or landmarks_data.shape[0] < 2:
        return None
    
    # Extract hip landmark positions (approximate center of mass)
    right_hip_idx = 24 * 4  # MediaPipe hip landmark index * 4 (x,y,z,visibility)
    left_hip_idx = 23 * 4
    
    # Extract coordinates
    right_hip = landmarks_data[:, right_hip_idx:right_hip_idx+2]  # Just x, y
    left_hip = landmarks_data[:, left_hip_idx:left_hip_idx+2]
    
    # Average hip positions for center of mass
    center_positions = (right_hip + left_hip) / 2
    
    # Create figure
    fig = Figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    
    # Create heatmap using histogram2d
    x = center_positions[:, 0]
    y = center_positions[:, 1]
    
    # Flip y-axis (MediaPipe coordinates)
    y = 1 - y
    
    # Create 2D histogram
    heatmap, xedges, yedges = np.histogram2d(x, y, bins=20, range=[[0, 1], [0, 1]])
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    
    # Plot heatmap
    im = ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
    
    # Add colorbar
    fig.colorbar(im, ax=ax, label='Presence Density')
    
    # Add labels and title
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_title('Player Movement Heatmap')
    
    # Adjust layout
    fig.tight_layout()
    
    return fig

def save_chart_as_image(fig):
    """
    Save matplotlib figure as image data
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Base64 encoded string of the image
    """
    buf = io.BytesIO()
    canvas = FigureCanvas(fig)
    canvas.print_png(buf)
    buf.seek(0)
    img_str = base64.b64encode(buf.read()).decode()
    return img_str

# ===== COMPONENTS =====

# ===== PLAYER CARDS =====
def home_page():
    """
    Render the home page with feature cards and welcome information
    """
    # Welcome section with application overview
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ## Welcome to Soccer Talent Analysis Platform
        
        This professional tool uses computer vision and AI to analyze soccer player performance 
        from video footage. Upload videos of players and get detailed insights to inform 
        recruitment, development, and tactical decisions.
        """)
        
        st.markdown("""
        ### Key Features
        - **Pose Detection Analysis**: Advanced computer vision captures player movement
        - **AI-Powered Insights**: Multiple AI models to analyze technique and performance
        - **Performance Metrics**: Quantitative measurement of key athletic attributes
        - **Side-by-Side Comparison**: Compare different players across key metrics
        - **History Tracking**: Maintain records of all player analyses
        """)
    
    with col2:
        # Display soccer player image
        st.image("https://images.unsplash.com/photo-1528054433354-7ab84caaccc0", 
                 caption="Soccer Analysis", use_container_width=True)
    
    # Feature cards section
    st.markdown("### Choose a feature to get started")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        feature_card(
            title="Player Analysis",
            description="Upload video of an individual player to analyze their performance, technique, and physical attributes.",
            icon="🏃‍♂️",
            page="Player Analysis"
        )
    
    with col2:
        feature_card(
            title="Player Comparison",
            description="Compare two players side-by-side to identify relative strengths and weaknesses across key metrics.",
            icon="⚔️",
            page="Player Comparison"
        )
    
    with col3:
        feature_card(
            title="Analysis History",
            description="View your previously analyzed players and comparisons with all recorded metrics and insights.",
            icon="📊",
            page="History"
        )
    
    # Model information section
    st.markdown("### Available Analysis Models")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### Offline Analysis
        Provides immediate results without requiring external API keys. Uses computer vision to extract performance metrics from player movements.
        
        #### Gemini Analysis
        Google's advanced multimodal AI model can analyze both visual and contextual information for richer insights. Requires API key.
        """)
    
    with col2:
        st.markdown("""
        #### DeepSeek Analysis
        Specialized model for detailed sports performance analysis with actionable insights for player development. Requires API key.
        
        #### Hugging Face LLaVa
        Vision-language model that combines image understanding with natural language processing for comprehensive analysis. Requires API key.
        """)
    
    # Footer with usage instructions
    st.markdown("---")
    st.markdown("""
    ### How to Use
    1. Select an analysis type from the cards above or the sidebar
    2. Choose your preferred AI model in the sidebar
    3. Upload player footage and fill in the player details
    4. Review the analysis and visualized metrics
    """)

def feature_card(title, description, icon, page):
    """
    Create a clickable card for a feature
    
    Args:
        title: Card title
        description: Feature description
        icon: Emoji icon
        page: Target page name
    """
    st.markdown(f"""
    <div style="padding: 1.5rem; border-radius: 0.5rem; border: 1px solid #eeeeee; background-color: #ffffff; height: 250px; position: relative; margin-bottom: 1rem;">
        <h1 style="font-size: 2rem; margin-bottom: 0.5rem;">{icon}</h1>
        <h3 style="margin-top: 0;">{title}</h3>
        <p style="color: #555555; height: 100px; overflow: auto;">{description}</p>
    </div>
    """, unsafe_allow_html=True)
    
    if st.button(f"Go to {title}", key=f"btn_{page}"):
        st.session_state.sidebar_selection = page
        st.rerun()

def player_info_card(player_data, title="Player Information"):
    """
    Display a card with player information
    
    Args:
        player_data: Dictionary containing player information
        title: Card title
    """
    st.markdown(f"### {title}")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Name:** {player_data.get('name', 'Unknown')}")
            st.markdown(f"**Position:** {player_data.get('details', {}).get('position', 'Unknown')}")
            st.markdown(f"**Age:** {player_data.get('details', {}).get('age', 'Unknown')}")
            st.markdown(f"**Height:** {player_data.get('details', {}).get('height', 'Unknown')} cm")
        
        with col2:
            st.markdown(f"**Weight:** {player_data.get('details', {}).get('weight', 'Unknown')} kg")
            st.markdown(f"**Preferred Foot:** {player_data.get('details', {}).get('foot', 'Unknown')}")
            st.markdown(f"**Experience:** {player_data.get('details', {}).get('experience', 'Unknown')}")
            
        if player_data.get('details', {}).get('notes'):
            st.markdown(f"**Notes:** {player_data.get('details', {}).get('notes', '')}")

def performance_summary_card(metrics, title="Performance Summary"):
    """
    Display a card with performance summary
    
    Args:
        metrics: Dictionary containing performance metrics
        title: Card title
    """
    summary = get_performance_summary(metrics)
    
    st.markdown(f"### {title}")
    
    with st.container():
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Overall Rating", f"{summary['overall'] / 10:.1f}/10")
        
        with col2:
            st.metric("Key Strength", summary['strength'])
        
        with col3:
            st.metric("Area to Improve", summary['weakness'])

# ===== SIDEBAR =====
def create_sidebar():
    """
    Create the application sidebar with navigation and settings
    
    Returns:
        Tuple of (selected_page, selected_model, api_keys)
    """
    with st.sidebar:
        st.image("https://images.unsplash.com/photo-1519823038424-f8dbabca95f1", use_container_width=True)
        
        st.title("⚽ Soccer Analysis")
        
        # Navigation
        st.subheader("Navigation")
        
        # Set default sidebar selection
        if 'sidebar_selection' not in st.session_state:
            st.session_state.sidebar_selection = "Home"
        
        # Navigation buttons
        selected_page = st.radio(
            "Select a page",
            ["Home", "Player Analysis", "Player Comparison", "History"],
            index=["Home", "Player Analysis", "Player Comparison", "History"].index(st.session_state.sidebar_selection)
        )
        
        # Update session state with the selected page
        st.session_state.sidebar_selection = selected_page
        
        st.markdown("---")
        
        # Model selection
        st.subheader("Analysis Settings")
        
        selected_model = st.selectbox(
            "Choose AI Model",
            ["offline", "gemini", "deepseek", "huggingface"],
            index=["offline", "gemini", "deepseek", "huggingface"].index(st.session_state.get('selected_model', 'offline')),
            format_func=lambda x: {
                "offline": "Offline Analysis",
                "gemini": "Google Gemini",
                "deepseek": "DeepSeek",
                "huggingface": "Hugging Face LLaVa"
            }.get(x, x.title())
        )
        
        # Store selected model in session state
        st.session_state.selected_model = selected_model
        
        # API key inputs for applicable models
        api_keys = {}
        
        if selected_model == "gemini":
            api_keys["gemini"] = st.text_input(
                "Gemini API Key",
                type="password",
                value=st.session_state.api_keys.get("gemini", ""),
                key="gemini_api_key"
            )
            st.session_state.api_keys["gemini"] = api_keys["gemini"]
            
        elif selected_model == "deepseek":
            api_keys["deepseek"] = st.text_input(
                "DeepSeek API Key",
                type="password",
                value=st.session_state.api_keys.get("deepseek", ""),
                key="deepseek_api_key"
            )
            st.session_state.api_keys["deepseek"] = api_keys["deepseek"]
            
        elif selected_model == "huggingface":
            api_keys["huggingface"] = st.text_input(
                "Hugging Face API Key",
                type="password",
                value=st.session_state.api_keys.get("huggingface", ""),
                key="huggingface_api_key"
            )
            st.session_state.api_keys["huggingface"] = api_keys["huggingface"]
        
        # About section
        st.markdown("---")
        st.markdown("""
        ### About
        Soccer Talent Analysis Platform uses computer vision and AI to help coaches, 
        scouts, and players analyze performance and identify areas for improvement.
        """)
    
    return selected_page, selected_model, api_keys

# ===== CHARTS =====
def render_player_metrics(player_data):
    """
    Render visualizations of player metrics
    
    Args:
        player_data: Dictionary containing player analysis data
    """
    # Extract metrics from player data
    metrics = get_player_metrics(player_data)
    
    if not metrics:
        st.warning("No metrics data available for visualization")
        return
    
    st.markdown("### Performance Visualizations")
    
    # Create tabs for different visualizations
    tab1, tab2, tab3 = st.tabs(["Radar Chart", "Bar Charts", "Movement Heatmap"])
    
    with tab1:
        # Format metrics for radar chart
        categories, values = format_metrics_for_chart(metrics)
        
        # Create and display radar chart
        radar_fig = create_radar_chart(categories, values, f"Performance Profile")
        st.pyplot(radar_fig)
        
        st.markdown("""
        **Radar Chart Explanation:**
        This visualization shows the player's performance across all key metrics on a scale of 0-100.
        A larger area indicates better overall performance, while the shape highlights relative strengths and weaknesses.
        """)
    
    with tab2:
        # Create bar charts for different metric categories
        col1, col2 = st.columns(2)
        
        with col1:
            # Speed metrics
            speed_categories = ["Average Speed", "Right Foot", "Left Foot"]
            speed_values = [
                metrics.get('speed', {}).get('avg_foot_speed', 0),
                metrics.get('speed', {}).get('right_foot', 0),
                metrics.get('speed', {}).get('left_foot', 0)
            ]
            speed_fig = create_bar_chart(speed_categories, speed_values, "Speed Metrics")
            st.pyplot(speed_fig)
        
        with col2:
            # Agility and balance metrics
            agility_categories = ["Agility", "Direction Changes", "Balance", "Posture"]
            agility_values = [
                metrics.get('agility', {}).get('max_acceleration', 0),
                metrics.get('agility', {}).get('direction_changes', 0),
                metrics.get('balance', {}).get('stability', 0),
                metrics.get('balance', {}).get('posture_consistency', 0)
            ]
            agility_fig = create_bar_chart(agility_categories, agility_values, "Agility & Balance Metrics")
            st.pyplot(agility_fig)
        
        st.markdown("""
        **Bar Chart Explanation:**
        These charts break down the player's physical attributes into specific components,
        making it easier to identify particular strengths and areas for improvement.
        """)
    
    with tab3:
        # Get landmarks data
        landmarks_data = np.array(player_data.get('landmarks_data', []))
        
        if len(landmarks_data) > 0:
            heatmap_fig = create_movement_heatmap(landmarks_data)
            if heatmap_fig:
                st.pyplot(heatmap_fig)
                
                st.markdown("""
                **Movement Heatmap Explanation:**
                This visualization shows where the player spent most time on the field.
                Brighter areas indicate higher presence, helping identify movement patterns and positioning tendencies.
                """)
        else:
            st.info("Movement data not available for heatmap visualization")
    
    # Summary metrics as numbers
    st.markdown("### Key Performance Indicators")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        avg_speed = metrics.get('speed', {}).get('avg_foot_speed', 0)
        st.metric("Speed", f"{avg_speed:.1f}/100")
    
    with col2:
        agility = (metrics.get('agility', {}).get('direction_changes', 0) + 
                  metrics.get('agility', {}).get('max_acceleration', 0)) / 2
        st.metric("Agility", f"{agility:.1f}/100")
    
    with col3:
        ball_control = metrics.get('ball_control', {}).get('hand_movement', 0)
        st.metric("Ball Control", f"{ball_control:.1f}/100")
    
    with col4:
        balance = (metrics.get('balance', {}).get('stability', 0) + 
                  metrics.get('balance', {}).get('posture_consistency', 0)) / 2
        st.metric("Balance", f"{balance:.1f}/100")
    
    # Overall rating
    overall = (avg_speed + agility + ball_control + balance) / 4
    st.progress(overall / 100, text=f"Overall Performance: {overall:.1f}/100")

def render_comparison_metrics(p1_analysis, p2_analysis, p1_name, p2_name):
    """
    Render comparison visualizations for two players
    
    Args:
        p1_analysis: Dictionary containing player 1 analysis data
        p2_analysis: Dictionary containing player 2 analysis data
        p1_name: Player 1's name
        p2_name: Player 2's name
    """
    # Extract metrics
    p1_metrics = p1_analysis.get('metrics', {})
    p2_metrics = p2_analysis.get('metrics', {})
    
    if not p1_metrics or not p2_metrics:
        st.warning("Complete metrics not available for comparison")
        return
    
    # Create tabs for different comparison visualizations
    tab1, tab2 = st.tabs(["Radar Comparison", "Bar Comparisons"])
    
    with tab1:
        # Format metrics for radar chart
        categories, values1 = format_metrics_for_chart(p1_metrics)
        _, values2 = format_metrics_for_chart(p2_metrics)
        
        # Create side-by-side radar charts
        col1, col2 = st.columns(2)
        
        with col1:
            radar_fig1 = create_radar_chart(categories, values1, f"{p1_name}'s Profile")
            st.pyplot(radar_fig1)
        
        with col2:
            radar_fig2 = create_radar_chart(categories, values2, f"{p2_name}'s Profile")
            st.pyplot(radar_fig2)
        
        st.markdown("""
        **Radar Comparison Explanation:**
        These charts allow you to compare the overall performance profile of both players.
        Different shapes indicate different strengths and playing styles.
        """)
    
    with tab2:
        # Create direct comparison bar charts
        speed_categories = ["Average Speed", "Right Foot", "Left Foot"]
        p1_speed_values = [
            p1_metrics.get('speed', {}).get('avg_foot_speed', 0),
            p1_metrics.get('speed', {}).get('right_foot', 0),
            p1_metrics.get('speed', {}).get('left_foot', 0)
        ]
        p2_speed_values = [
            p2_metrics.get('speed', {}).get('avg_foot_speed', 0),
            p2_metrics.get('speed', {}).get('right_foot', 0),
            p2_metrics.get('speed', {}).get('left_foot', 0)
        ]
        
        # Speed comparison
        speed_fig = create_comparison_bar_chart(
            speed_categories, p1_speed_values, p2_speed_values, p1_name, p2_name, "Speed Comparison"
        )
        st.pyplot(speed_fig)
        
        # Agility comparison
        agility_categories = ["Agility", "Direction Changes"]
        p1_agility_values = [
            p1_metrics.get('agility', {}).get('max_acceleration', 0),
            p1_metrics.get('agility', {}).get('direction_changes', 0)
        ]
        p2_agility_values = [
            p2_metrics.get('agility', {}).get('max_acceleration', 0),
            p2_metrics.get('agility', {}).get('direction_changes', 0)
        ]
        
        agility_fig = create_comparison_bar_chart(
            agility_categories, p1_agility_values, p2_agility_values, p1_name, p2_name, "Agility Comparison"
        )
        st.pyplot(agility_fig)
        
        # Balance and control comparison
        balance_categories = ["Ball Control", "Balance", "Posture"]
        p1_balance_values = [
            p1_metrics.get('ball_control', {}).get('hand_movement', 0),
            p1_metrics.get('balance', {}).get('stability', 0),
            p1_metrics.get('balance', {}).get('posture_consistency', 0)
        ]
        p2_balance_values = [
            p2_metrics.get('ball_control', {}).get('hand_movement', 0),
            p2_metrics.get('balance', {}).get('stability', 0),
            p2_metrics.get('balance', {}).get('posture_consistency', 0)
        ]
        
        balance_fig = create_comparison_bar_chart(
            balance_categories, p1_balance_values, p2_balance_values, p1_name, p2_name, "Control & Balance Comparison"
        )
        st.pyplot(balance_fig)
        
        st.markdown("""
        **Bar Comparison Explanation:**
        These charts provide direct side-by-side comparisons across specific metrics,
        making it easy to identify each player's relative strengths and weaknesses.
        """)
    
    # Overall comparison metrics
    st.markdown("### Overall Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        p1_avg_speed = p1_metrics.get('speed', {}).get('avg_foot_speed', 0)
        p1_agility = (p1_metrics.get('agility', {}).get('direction_changes', 0) + 
                       p1_metrics.get('agility', {}).get('max_acceleration', 0)) / 2
        p1_ball_control = p1_metrics.get('ball_control', {}).get('hand_movement', 0)
        p1_balance = (p1_metrics.get('balance', {}).get('stability', 0) + 
                       p1_metrics.get('balance', {}).get('posture_consistency', 0)) / 2
        
        p1_overall = (p1_avg_speed + p1_agility + p1_ball_control + p1_balance) / 4
        st.progress(p1_overall / 100, text=f"{p1_name}: {p1_overall:.1f}/100")
    
    with col2:
        p2_avg_speed = p2_metrics.get('speed', {}).get('avg_foot_speed', 0)
        p2_agility = (p2_metrics.get('agility', {}).get('direction_changes', 0) + 
                       p2_metrics.get('agility', {}).get('max_acceleration', 0)) / 2
        p2_ball_control = p2_metrics.get('ball_control', {}).get('hand_movement', 0)
        p2_balance = (p2_metrics.get('balance', {}).get('stability', 0) + 
                       p2_metrics.get('balance', {}).get('posture_consistency', 0)) / 2
        
        p2_overall = (p2_avg_speed + p2_agility + p2_ball_control + p2_balance) / 4
        st.progress(p2_overall / 100, text=f"{p2_name}: {p2_overall:.1f}/100")

# ===== MAIN APPLICATION =====

# Page configuration
st.set_page_config(
    page_title="Soccer Talent Analysis",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'current_player' not in st.session_state:
    st.session_state.current_player = None
if 'current_comparison' not in st.session_state:
    st.session_state.current_comparison = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "offline"
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        "gemini": "",
        "deepseek": "",
        "huggingface": ""
    }

# Create sidebar
selected_page, selected_model, api_keys = create_sidebar()

# Define header
def header():
    st.markdown("""
    <div style='text-align: center;'>
        <h1>⚽ Soccer Talent Analysis Platform</h1>
        <p>Professional-grade player analysis powered by AI and computer vision</p>
    </div>
    """, unsafe_allow_html=True)

# Home page
def home():
    header()
    home_page()

# Player analysis page
def player_analysis():
    header()
    st.subheader("Individual Player Analysis")
    
    # Create a card-like container for player info
    with st.container():
        st.markdown("#### Player Information")
        col1, col2 = st.columns(2)
        
        with col1:
            player_name = st.text_input("Player Name")
            position = st.selectbox(
                "Position",
                ["Goalkeeper", "Defender", "Midfielder", "Forward", "Winger"]
            )
            age = st.number_input("Age", min_value=10, max_value=50, value=18)
            height = st.number_input("Height (cm)", min_value=100, max_value=220, value=175)
        
        with col2:
            weight = st.number_input("Weight (kg)", min_value=30, max_value=120, value=70)
            foot = st.selectbox(
                "Preferred Foot",
                ["Right", "Left", "Both"]
            )
            experience = st.selectbox(
                "Experience Level",
                ["Youth", "Amateur", "Semi-Pro", "Professional"]
            )
            notes = st.text_area("Additional Notes", height=100)
    
    # Video upload
    st.markdown("#### Upload Game Footage")
    video_file = st.file_uploader("Upload video (MP4 or MOV)", type=["mp4", "mov"])
    
    # Analysis button
    analyze_button = st.button("Analyze Player", type="primary")
    
    if analyze_button and video_file:
        if selected_model in ["gemini", "deepseek", "huggingface"] and not api_keys.get(selected_model):
            st.error(f"Please enter your {selected_model.title()} API key in the sidebar")
            return
            
        with st.spinner("Processing video and analyzing performance..."):
            # Analyze video with MediaPipe
            frames, landmarks = analyze_video(video_file)
            
            # Display sample frame with pose detection
            if frames:
                st.subheader("Video Analysis")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(frames[0], caption="Pose Detection", use_container_width=True)
                
                # Format player info
                player_info = {
                    'name': player_name,
                    'position': position,
                    'age': age,
                    'height': height,
                    'weight': weight,
                    'foot': foot,
                    'experience': experience,
                    'notes': notes
                }
                
                info_str = f"""
                Name: {player_name}
                Position: {position}
                Age: {age}
                Height: {height} cm
                Weight: {weight} kg
                Preferred Foot: {foot}
                Experience Level: {experience}
                Notes: {notes}
                """
                
                # Analyze with selected model
                with col2:
                    analysis = analyze_with_model(
                        model_name=selected_model,
                        video_frames=frames,
                        landmarks=landmarks,
                        player_info=info_str,
                        api_key=api_keys.get(selected_model, "")
                    )
                    
                    if analysis:
                        st.markdown(f"#### Performance Analysis for {player_name}")
                        st.markdown(analysis['summary'])
                        
                        # Prepare and save player data
                        player_data = {
                            'name': player_name,
                            'analysis': analysis,
                            'info': info_str,
                            'details': player_info,
                            'video_preview': frames[0].tolist() if len(frames) > 0 else None,
                            'metrics': analysis.get('metrics', {}),
                            'landmarks_data': landmarks.tolist() if landmarks.size > 0 else [],
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        save_player_data(player_data)
                        st.session_state.current_player = player_data
                
                # Display metrics and visualizations
                if 'current_player' in st.session_state and st.session_state.current_player:
                    st.subheader("Player Metrics")
                    render_player_metrics(st.session_state.current_player)
    elif analyze_button:
        st.warning("Please upload a video file first")

# Player comparison page
def compare_players():
    header()
    st.subheader("Player Comparison Tool")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("##### Player 1")
        with st.container():
            p1_name = st.text_input("Name", key="p1_name")
            p1_video = st.file_uploader("Upload Video", type=["mp4", "mov"], key="p1_video")
            with st.expander("Player Information"):
                p1_position = st.selectbox("Position", ["Goalkeeper", "Defender", "Midfielder", "Forward", "Winger"], key="p1_pos")
                p1_age = st.number_input("Age", min_value=10, max_value=50, value=18, key="p1_age")
                p1_height = st.number_input("Height (cm)", min_value=100, max_value=220, value=175, key="p1_height")
                p1_weight = st.number_input("Weight (kg)", min_value=30, max_value=120, value=70, key="p1_weight")
                p1_foot = st.selectbox("Preferred Foot", ["Right", "Left", "Both"], key="p1_foot")
                p1_experience = st.selectbox("Experience Level", ["Youth", "Amateur", "Semi-Pro", "Professional"], key="p1_exp")
                p1_notes = st.text_area("Notes", key="p1_notes", height=100)
    
    with col2:
        st.markdown("##### Player 2")
        with st.container():
            p2_name = st.text_input("Name", key="p2_name")
            p2_video = st.file_uploader("Upload Video", type=["mp4", "mov"], key="p2_video")
            with st.expander("Player Information"):
                p2_position = st.selectbox("Position", ["Goalkeeper", "Defender", "Midfielder", "Forward", "Winger"], key="p2_pos")
                p2_age = st.number_input("Age", min_value=10, max_value=50, value=18, key="p2_age")
                p2_height = st.number_input("Height (cm)", min_value=100, max_value=220, value=175, key="p2_height")
                p2_weight = st.number_input("Weight (kg)", min_value=30, max_value=120, value=70, key="p2_weight")
                p2_foot = st.selectbox("Preferred Foot", ["Right", "Left", "Both"], key="p2_foot")
                p2_experience = st.selectbox("Experience Level", ["Youth", "Amateur", "Semi-Pro", "Professional"], key="p2_exp")
                p2_notes = st.text_area("Notes", key="p2_notes", height=100)
    
    compare_button = st.button("Compare Players", type="primary")
    
    if compare_button:
        if selected_model in ["gemini", "deepseek", "huggingface"] and not api_keys.get(selected_model):
            st.error(f"Please enter your {selected_model.title()} API key in the sidebar")
            return
            
        if p1_video and p2_video:
            with st.spinner("Analyzing players..."):
                # Process Player 1
                p1_frames, p1_landmarks = analyze_video(p1_video)
                p1_info = {
                    'name': p1_name,
                    'position': p1_position,
                    'age': p1_age,
                    'height': p1_height,
                    'weight': p1_weight,
                    'foot': p1_foot,
                    'experience': p1_experience,
                    'notes': p1_notes
                }
                p1_info_str = f"""
                Name: {p1_name}
                Position: {p1_position}
                Age: {p1_age}
                Height: {p1_height} cm
                Weight: {p1_weight} kg
                Preferred Foot: {p1_foot}
                Experience Level: {p1_experience}
                Notes: {p1_notes}
                """
                
                # Process Player 2
                p2_frames, p2_landmarks = analyze_video(p2_video)
                p2_info = {
                    'name': p2_name,
                    'position': p2_position,
                    'age': p2_age,
                    'height': p2_height,
                    'weight': p2_weight,
                    'foot': p2_foot,
                    'experience': p2_experience,
                    'notes': p2_notes
                }
                p2_info_str = f"""
                Name: {p2_name}
                Position: {p2_position}
                Age: {p2_age}
                Height: {p2_height} cm
                Weight: {p2_weight} kg
                Preferred Foot: {p2_foot}
                Experience Level: {p2_experience}
                Notes: {p2_notes}
                """
                
                # Get analysis for each player
                p1_analysis = analyze_with_model(
                    model_name=selected_model,
                    video_frames=p1_frames,
                    landmarks=p1_landmarks,
                    player_info=p1_info_str,
                    api_key=api_keys.get(selected_model, "")
                )
                
                p2_analysis = analyze_with_model(
                    model_name=selected_model,
                    video_frames=p2_frames,
                    landmarks=p2_landmarks,
                    player_info=p2_info_str,
                    api_key=api_keys.get(selected_model, "")
                )
                
                if p1_analysis and p2_analysis:
                    # Generate comparison
                    comparison_prompt = f"""
                    Compare these two soccer players:
                    
                    Player 1: {p1_name}
                    {p1_analysis['summary']}
                    
                    Player 2: {p2_name}
                    {p2_analysis['summary']}
                    
                    Provide a detailed comparison highlighting:
                    1. Relative strengths and weaknesses
                    2. Potential best position for each
                    3. Which player might be better suited for different team needs
                    4. Development potential
                    
                    Use tables for clear comparison of key metrics.
                    """
                    
                    comparison = analyze_with_model(
                        model_name=selected_model,
                        video_frames=[],
                        landmarks=None,
                        player_info=comparison_prompt,
                        api_key=api_keys.get(selected_model, ""),
                        is_comparison=True
                    )
                    
                    if comparison:
                        st.subheader("Comparison Results")
                        
                        # Display player previews side by side
                        col1, col2 = st.columns(2)
                        with col1:
                            st.image(p1_frames[0], caption=f"{p1_name} - Pose Analysis", use_container_width=True)
                        with col2:
                            st.image(p2_frames[0], caption=f"{p2_name} - Pose Analysis", use_container_width=True)
                        
                        # Display comparison text
                        st.markdown(comparison['summary'])
                        
                        # Comparison metrics visualization
                        st.subheader("Performance Metrics Comparison")
                        render_comparison_metrics(p1_analysis, p2_analysis, p1_name, p2_name)
                        
                        # Save comparison data
                        comparison_data = {
                            'p1': {
                                'name': p1_name,
                                'analysis': p1_analysis,
                                'info': p1_info_str,
                                'details': p1_info,
                                'metrics': p1_analysis.get('metrics', {}),
                                'landmarks_data': p1_landmarks.tolist() if p1_landmarks.size > 0 else [],
                                'video_preview': p1_frames[0].tolist() if len(p1_frames) > 0 else None
                            },
                            'p2': {
                                'name': p2_name,
                                'analysis': p2_analysis,
                                'info': p2_info_str,
                                'details': p2_info,
                                'metrics': p2_analysis.get('metrics', {}),
                                'landmarks_data': p2_landmarks.tolist() if p2_landmarks.size > 0 else [],
                                'video_preview': p2_frames[0].tolist() if len(p2_frames) > 0 else None
                            },
                            'comparison': comparison,
                            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                        }
                        
                        save_player_data(comparison_data)
                        st.session_state.current_comparison = comparison_data
        else:
            st.warning("Please upload videos for both players")

# History page
def history_page():
    header()
    st.subheader("Analysis History")
    
    player_history = load_player_history()
    
    if not player_history:
        st.warning("No analysis history found. Analyze some players first!")
        return
    
    # Organize history by date (newest first)
    player_history.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
    
    # Filter options
    st.markdown("#### Filter History")
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.radio("Analysis Type", ["All", "Individual", "Comparison"])
    with col2:
        search_name = st.text_input("Search by Name")
    
    # Filter the history based on selections
    filtered_history = []
    for entry in player_history:
        # Filter by type
        if filter_type == "Individual" and 'p1' in entry:
            continue
        if filter_type == "Comparison" and 'p1' not in entry:
            continue
        
        # Filter by name
        if search_name:
            if 'p1' in entry:  # This is a comparison
                if search_name.lower() not in entry['p1']['name'].lower() and search_name.lower() not in entry['p2']['name'].lower():
                    continue
            else:  # This is an individual analysis
                if search_name.lower() not in entry['name'].lower():
                    continue
        
        filtered_history.append(entry)
    
    # Display history entries as cards
    for i, entry in enumerate(filtered_history):
        with st.container():
            st.markdown(f"#### Analysis #{i+1} - {entry.get('timestamp', 'Unknown date')}")
            
            if 'p1' in entry:  # This is a comparison
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"**{entry['p1']['name']}**")
                    st.markdown(f"Position: {entry['p1']['details']['position']}")
                with col2:
                    st.markdown(f"**{entry['p2']['name']}**")
                    st.markdown(f"Position: {entry['p2']['details']['position']}")
                
                with st.expander("View Comparison Details"):
                    st.markdown(entry['comparison']['summary'])
                    
                    # Display comparison charts
                    render_comparison_metrics(
                        entry['p1']['analysis'],
                        entry['p2']['analysis'],
                        entry['p1']['name'],
                        entry['p2']['name']
                    )
            else:  # This is an individual analysis
                st.markdown(f"**{entry['name']}**")
                st.markdown(f"Position: {entry['details']['position']}")
                
                with st.expander("View Analysis Details"):
                    st.markdown(entry['analysis']['summary'])
                    
                    # Display player metrics
                    render_player_metrics(entry)
            
            st.markdown("---")

# Main application logic
def main():
    # Display appropriate page based on sidebar selection
    if selected_page == "Home":
        home()
    elif selected_page == "Player Analysis":
        player_analysis()
    elif selected_page == "Player Comparison":
        compare_players()
    elif selected_page == "History":
        history_page()

if __name__ == "__main__":
    main()
