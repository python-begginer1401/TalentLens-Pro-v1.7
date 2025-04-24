# TalentLens Pro v1.7 Scout smarter – not harder!

![Soccer Analysis](https://images.unsplash.com/photo-1519823038424-f8dbabca95f1?auto=format&w=800)

## Overview

Soccer Talent Analysis Platform is a sophisticated application designed for coaches, scouts, and players to analyze soccer performance using computer vision and AI. The platform processes video footage to extract meaningful metrics and provide actionable insights for player development.

## Features

### 🏃‍♂️ Individual Player Analysis
Upload videos of individual players to analyze their performance, technique, and physical attributes. The platform uses computer vision to extract motion data and provides detailed metrics across key performance areas.

### ⚔️ Player Comparison
Compare two players side-by-side to identify relative strengths and weaknesses across all performance metrics. Visualize differences in speed, agility, balance, and technique through intuitive comparative charts.

### 📊 Performance History
Keep track of all previously analyzed players and comparisons with a comprehensive history feature. Review past analyses, track player development over time, and export metrics for external use.

## AI Analysis Models

The platform offers multiple AI models for analysis:

- **Offline Analysis**: Provides immediate results without requiring external API keys. Uses computer vision to extract performance metrics from player movements.

- **Google Gemini**: Google's advanced multimodal AI model can analyze both visual and contextual information for richer insights. Requires API key.

- **DeepSeek**: Specialized model for detailed sports performance analysis with actionable insights for player development. Requires API key.

- **Hugging Face LLaVa**: Vision-language model that combines image understanding with natural language processing for comprehensive analysis. Requires API key.

## Performance Metrics

The platform calculates and visualizes various performance metrics:

- **Speed & Movement**: Average speed, left/right foot speed, movement patterns
- **Agility**: Acceleration, direction changes, reaction time
- **Balance**: Stability, posture consistency, weight distribution
- **Ball Control**: Hand-eye coordination, footwork precision
- **Overall Performance**: Comprehensive rating based on all metrics

## Visualizations

- **Radar Charts**: Overall performance profile across all metrics
- **Bar Charts**: Detailed breakdown of specific metric categories
- **Movement Heatmaps**: Visual representation of player positioning and movement patterns
- **Comparison Charts**: Side-by-side comparison of two players' metrics

## Technical Details

### Requirements

- Python 3.8+
- Streamlit
- OpenCV
- MediaPipe
- NumPy
- Matplotlib
- Pandas
- API Keys (for external AI models)

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/soccer-analysis-platform.git
   cd soccer-analysis-platform
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   streamlit run app.py
   ```

### API Keys

To use the external AI models, you'll need to obtain the following API keys:

- **Google Gemini**: Available from [Google AI Studio](https://ai.google.dev/)
- **DeepSeek**: Available from [DeepSeek AI](https://www.deepseek.ai/)
- **Hugging Face**: Available from [Hugging Face](https://huggingface.co/settings/tokens)

## Usage Guide

1. **Select Analysis Type**: Choose between individual player analysis, player comparison, or viewing analysis history.
2. **Select AI Model**: Choose your preferred analysis model and enter API keys if required.
3. **Upload Video**: Upload footage of soccer players in action (MP4 or MOV format).
4. **Enter Player Details**: Provide player information like name, position, age, and other relevant details.
5. **Review Analysis**: Explore the comprehensive analysis with performance metrics, visualizations, and AI-generated insights.

## Acknowledgments

- [MediaPipe](https://github.com/google/mediapipe) for pose detection
- [Streamlit](https://streamlit.io/) for the web application framework
- [OpenCV](https://opencv.org/) for video processing
- [Matplotlib](https://matplotlib.org/) for data visualization

---

Developed with ❤️ for the soccer community by the TalentLens team
