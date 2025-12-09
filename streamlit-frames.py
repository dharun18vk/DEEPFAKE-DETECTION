import os
import warnings

# Suppress all warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

import cv2
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from tqdm import tqdm
import tempfile
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd

# Set matplotlib to avoid GUI warnings
plt.set_loglevel('warning')

# =============================================================================
# MODEL PATHS CONFIGURATION - Define your model paths here
# =============================================================================
MODEL_PATHS = {
    "FF++,celeb,image Model": "DEEPFAKE MODELS/best_stage3_ffpp_frames.pth",
    "Celeb-DF Fine-tuned Model": "DEEPFAKE MODELS/best_celebf_finetuned.pth", 
    "celeb,image Model": "DEEPFAKE MODELS/2.5-df.pth",
    "celeb DF model(training ongoing)":"DEEPFAKE MODELS/best_celebf_finetuned.pth",
    "image model":"DEEPFAKE MODELS/1-df.pth",
    "multi fine tuned model(under training)":"DEEPFAKE MODELS/1.5-df.pth",    
    "celeb model": "DEEPFAKE MODELS/0-df.pth",
    "Face trained model(new)":r"D:\deepfake_train\new_deepfake_model\best_working_model.pth",
    "New face df model":"new_deepfake_model/best_model.pth",
    "new xception model":"xception_deepfake_model/best_face_model.pth",
    "level 1 fine tuned xception":"DEEPFAKE MODELS/fine_tuned_xception_model/best_fine_tuned_model.pth",
    "level 2 fine tuned xception":"DEEPFAKE MODELS/progressive_fine_tuned_model/2nd_tuned_xception_model.pth",
    "combined dataset model(new)":"xception_finetuned_model/best_xception_finetuned.pth",
    "efficient net B4 model":"optimized_efficientnet_b4_model/best_B4_model.pth",
    "Final trained B4 model":"checkpoints/best_model.pth" 
}
# =============================================================================

# Custom JSON encoder to handle numpy types and other non-serializable objects
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.datetime64, np.timedelta64)):
            return str(obj)
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        elif isinstance(obj, (bytes, bytearray)):
            return obj.decode('utf-8', errors='ignore')
        return super(NumpyEncoder, self).default(obj)

class FrameExtractor:
    def __init__(self, output_dir="extracted_frames"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def extract_frames(self, video_path, frames_per_second=1):
        with st.status("🎬 **Extracting frames from video...**", expanded=True) as status:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                st.error(f"❌ Error opening video file: {video_path}")
                return []
            
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps
            
            # Video info in a modern card
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🎯 FPS", f"{fps:.2f}")
            with col2:
                st.metric("📊 Total Frames", f"{total_frames}")
            with col3:
                st.metric("⏱️ Duration", f"{duration:.2f}s")
            
            frame_interval = max(1, int(fps / frames_per_second))
            extracted_frames = []
            frame_count = 0
            
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            video_frames_dir = os.path.join(self.output_dir, video_name)
            os.makedirs(video_frames_dir, exist_ok=True)
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_count % frame_interval == 0:
                    frame_filename = f"frame_{frame_count:06d}.jpg"
                    frame_path = os.path.join(video_frames_dir, frame_filename)
                    cv2.imwrite(frame_path, frame)
                    extracted_frames.append({
                        'path': frame_path,
                        'frame_number': int(frame_count),
                        'timestamp': float(frame_count / fps)
                    })
                
                frame_count += 1
                if frame_count % 100 == 0:
                    progress = min(float(frame_count / total_frames), 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            progress_bar.progress(1.0)
            status.update(label=f"✅ **Frame Extraction Complete** - Extracted {len(extracted_frames)} frames", state="complete")
        return extracted_frames

class DeepFakeDetector:
    def __init__(self, model_path):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
    
    def load_model(self, model_path):
        with st.status("🤖 **Loading deepfake detection model...**", expanded=True) as status:
            if not os.path.exists(model_path):
                st.error(f"❌ Model file not found: {model_path}")
                return None
            
            model = models.resnet50(pretrained=False)
            num_features = model.fc.in_features
            
            model.fc = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )
            
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                
                # Handle different checkpoint formats
                if 'model_state_dict' in checkpoint:
                    state_dict = checkpoint['model_state_dict']
                elif 'state_dict' in checkpoint:
                    state_dict = checkpoint['state_dict']
                else:
                    state_dict = checkpoint
                
                # Load with strict=False to handle architecture differences
                model.load_state_dict(state_dict, strict=False)
                
                # Show model info
                total_params = sum(p.numel() for p in model.parameters())
                
                status.update(label=f"✅ **Model Loaded Successfully** - {total_params:,} parameters", state="complete")
                
                # Model info cards
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.info(f"🧠 **Architecture:** ResNet50")
                with col2:
                    st.info(f"⚡ **Device:** {'GPU' if torch.cuda.is_available() else 'CPU'}")
                with col3:
                    st.info(f"📊 **Parameters:** {total_params:,}")
                
            except Exception as e:
                st.error(f"❌ Error loading model: {e}")
                return None
            
            model.to(self.device)
            model.eval()
            return model
    
    def get_transform(self):
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def predict_frame(self, frame_path):
        try:
            image = Image.open(frame_path).convert('RGB')
            image_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.model(image_tensor)
                probability = output.item()
                prediction = "FAKE" if probability > 0.5 else "REAL"
                confidence = probability if prediction == "FAKE" else 1 - probability
            
            return {
                'prediction': prediction,
                'confidence': float(confidence),
                'fake_probability': float(probability),
                'real_probability': float(1 - probability),
                'frame_path': frame_path
            }
        
        except Exception as e:
            st.warning(f"⚠️ Error processing frame {frame_path}: {str(e)}")
            return None
    
    def predict_multiple_frames(self, frame_paths):
        with st.status("🔍 **Analyzing frames for deepfake detection...**", expanded=True) as status:
            results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, frame_info in enumerate(frame_paths):
                result = self.predict_frame(frame_info['path'])
                if result:
                    # Add frame metadata to result
                    result.update({
                        'frame_number': frame_info['frame_number'],
                        'timestamp': frame_info['timestamp']
                    })
                    results.append(result)
                
                progress = float((i + 1) / len(frame_paths))
                progress_bar.progress(progress)
                if (i + 1) % 10 == 0 or (i + 1) == len(frame_paths):
                    status_text.text(f"🧪 Analyzed {i+1}/{len(frame_paths)} frames")
            
            # Analysis summary
            if results:
                fake_count = sum(1 for r in results if r['prediction'] == 'FAKE')
                real_count = len(results) - fake_count
                
                col1, col2 = st.columns(2)
                with col1:
                    st.success(f"✅ **Real Frames:** {real_count}")
                with col2:
                    st.error(f"❌ **Fake Frames:** {fake_count}")
            
            status.update(label=f"✅ **Analysis Complete** - Processed {len(results)} frames", state="complete")
        return results

class DeepFakeAnalysisPipeline:
    def __init__(self, deepfake_model_path):
        self.frame_extractor = FrameExtractor()
        self.deepfake_detector = DeepFakeDetector(deepfake_model_path)
        self.results_dir = "analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_video(self, video_path, frames_per_second=1):
        # Modern header with gradient
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">🚀 Deepfake Analysis Pipeline</h1>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Frame-based Deepfake Detection System</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Check if model is loaded
        if self.deepfake_detector.model is None:
            st.error("❌ Deepfake model failed to load. Please check the model path.")
            return None
        
        # Step 1: Extract frames
        with st.container():
            st.markdown("### 🎬 STEP 1: Frame Extraction")
            frames = self.frame_extractor.extract_frames(video_path, frames_per_second)
            if not frames:
                st.error("❌ No frames extracted. Exiting.")
                return None
        
        # Step 2: Deepfake analysis
        with st.container():
            st.markdown("### 🔍 STEP 2: Deepfake Analysis")
            predictions = self.deepfake_detector.predict_multiple_frames(frames)
            
            if not predictions:
                st.error("❌ No frames were successfully analyzed.")
                return None
        
        # Step 3: Generate report
        with st.container():
            st.markdown("### 📊 STEP 3: Generating Analysis Report")
            analysis_report = self.generate_report(predictions, video_path)
        
        # Step 4: Visualize results
        with st.container():
            st.markdown("### 📈 STEP 4: Comprehensive Visualizations")
            self.create_enhanced_visualizations(predictions, analysis_report)
        
        return analysis_report
    
    def generate_report(self, predictions, video_path):
        analyzed_frames = [pred for pred in predictions if 'prediction' in pred]
        total_frames = len(analyzed_frames)
        
        if total_frames == 0:
            st.error("❌ No frames were successfully analyzed.")
            return None
        
        fake_frames = sum(1 for pred in analyzed_frames if pred['prediction'] == 'FAKE')
        real_frames = total_frames - fake_frames
        
        fake_confidence_avg = float(np.mean([pred['confidence'] for pred in analyzed_frames if pred['prediction'] == 'FAKE'])) if fake_frames > 0 else 0.0
        real_confidence_avg = float(np.mean([pred['confidence'] for pred in analyzed_frames if pred['prediction'] == 'REAL'])) if real_frames > 0 else 0.0
        
        report = {
            'video_path': video_path,
            'analysis_timestamp': str(np.datetime64('now')),
            'total_frames_analyzed': int(total_frames),
            'fake_frames_detected': int(fake_frames),
            'real_frames_detected': int(real_frames),
            'fake_percentage': float((fake_frames / total_frames * 100) if total_frames > 0 else 0),
            'average_fake_confidence': float(fake_confidence_avg),
            'average_real_confidence': float(real_confidence_avg),
            'overall_verdict': "LIKELY FAKE" if fake_frames > real_frames else "LIKELY REAL",
            'confidence_score': float(max(fake_confidence_avg, real_confidence_avg)),
            'detailed_analysis': analyzed_frames
        }
        
        report_path = os.path.join(self.results_dir, f"analysis_report_{os.path.basename(video_path)}.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            st.success(f"✅ Analysis report saved: {report_path}")
        except Exception as e:
            st.error(f"❌ Error saving report: {e}")
            return None
        
        return report
    
    def create_enhanced_visualizations(self, predictions, report):
        if not predictions or 'prediction' not in predictions[0]:
            st.warning("⚠️ No valid prediction data for visualization")
            return
        
        analyzed_frames = [pred for pred in predictions if 'prediction' in pred]
        if not analyzed_frames:
            st.warning("⚠️ No analyzed frames for visualization")
            return
        
        # Create DataFrame for easier plotting
        df = pd.DataFrame(analyzed_frames)
        
        # 1. Pie Chart - Results Distribution
        st.markdown("### 📊 Results Distribution")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_pie = px.pie(
                names=['Real', 'Fake'],
                values=[report['real_frames_detected'], report['fake_frames_detected']],
                title='Frame Analysis Results',
                color=['Real', 'Fake'],
                color_discrete_map={'Real': '#4ECDC4', 'Fake': '#FF6B6B'}
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # 2. Bar Chart - Confidence Comparison
        with col2:
            confidence_data = {
                'Type': ['Average Real Confidence', 'Average Fake Confidence', 'Overall Confidence'],
                'Value': [
                    report['average_real_confidence'],
                    report['average_fake_confidence'],
                    report['confidence_score']
                ]
            }
            fig_bar = px.bar(
                confidence_data,
                x='Type',
                y='Value',
                title='Confidence Scores Comparison',
                color='Type',
                color_discrete_map={
                    'Average Real Confidence': '#4ECDC4',
                    'Average Fake Confidence': '#FF6B6B',
                    'Overall Confidence': '#45B7D1'
                }
            )
            fig_bar.update_layout(yaxis_range=[0, 1])
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # 3. Histogram - Fake Probability Distribution
        st.markdown("### 📈 Probability Distributions")
        col3, col4 = st.columns(2)
        
        with col3:
            fig_hist = px.histogram(
                df,
                x='fake_probability',
                nbins=20,
                title='Fake Probability Distribution',
                color_discrete_sequence=['#FF6B6B']
            )
            fig_hist.update_layout(
                xaxis_title='Fake Probability',
                yaxis_title='Number of Frames'
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        # 4. Timeline Analysis
        with col4:
            fig_timeline = px.scatter(
                df,
                x='timestamp',
                y='fake_probability',
                color='prediction',
                title='Deepfake Probability Timeline',
                color_discrete_map={'REAL': '#4ECDC4', 'FAKE': '#FF6B6B'},
                size='confidence',
                hover_data=['frame_number']
            )
            fig_timeline.update_layout(
                xaxis_title='Time (seconds)',
                yaxis_title='Fake Probability'
            )
            st.plotly_chart(fig_timeline, use_container_width=True)
        
        # 5. Cumulative Analysis
        st.markdown("### 📊 Cumulative Analysis")
        df_sorted = df.sort_values('timestamp')
        df_sorted['cumulative_fake'] = (df_sorted['prediction'] == 'FAKE').cumsum()
        df_sorted['cumulative_real'] = (df_sorted['prediction'] == 'REAL').cumsum()
        
        fig_cumulative = go.Figure()
        fig_cumulative.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['cumulative_fake'],
            name='Cumulative Fake Frames',
            line=dict(color='#FF6B6B', width=3)
        ))
        fig_cumulative.add_trace(go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['cumulative_real'],
            name='Cumulative Real Frames',
            line=dict(color='#4ECDC4', width=3)
        ))
        fig_cumulative.update_layout(
            title='Cumulative Frame Analysis Over Time',
            xaxis_title='Time (seconds)',
            yaxis_title='Cumulative Frame Count'
        )
        st.plotly_chart(fig_cumulative, use_container_width=True)
        
        # 6. Confidence Distribution by Prediction
        st.markdown("### 🎯 Confidence Analysis")
        fig_violin = px.violin(
            df,
            y='confidence',
            x='prediction',
            color='prediction',
            box=True,
            points="all",
            title='Confidence Distribution by Prediction',
            color_discrete_map={'REAL': '#4ECDC4', 'FAKE': '#FF6B6B'}
        )
        st.plotly_chart(fig_violin, use_container_width=True)
    
    def print_summary(self, report):
        if report is None:
            st.error("❌ No report available to display")
            return
            
        # Enhanced summary with modern cards
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: white; text-align: center; margin: 0; font-size: 2rem;">🎯 DEEPFAKE ANALYSIS SUMMARY</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Main metrics in modern cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label="📹 Video File",
                value=os.path.basename(report['video_path'])[:15] + "..." if len(os.path.basename(report['video_path'])) > 15 else os.path.basename(report['video_path']),
                delta=None
            )
        
        with col2:
            st.metric(
                label="🎬 Frames Analyzed",
                value=int(report['total_frames_analyzed']),
                delta=None
            )
        
        with col3:
            st.metric(
                label="✅ Real Frames",
                value=int(report['real_frames_detected']),
                delta=f"+{int(report['real_frames_detected'])}" if report['real_frames_detected'] > 0 else "0"
            )
        
        with col4:
            st.metric(
                label="❌ Fake Frames",
                value=int(report['fake_frames_detected']),
                delta=f"+{int(report['fake_frames_detected'])}" if report['fake_frames_detected'] > 0 else "0",
                delta_color="inverse"
            )
        
        # Confidence metrics
        col5, col6, col7 = st.columns(3)
        
        with col5:
            st.metric(
                label="📊 Fake Percentage",
                value=f"{float(report['fake_percentage']):.1f}%",
                delta=None
            )
        
        with col6:
            st.metric(
                label="💪 Overall Confidence",
                value=f"{float(report['confidence_score']):.3f}",
                delta=None
            )
        
        with col7:
            avg_conf = (float(report['average_real_confidence']) + float(report['average_fake_confidence'])) / 2
            st.metric(
                label="⚡ Average Confidence",
                value=f"{avg_conf:.3f}",
                delta=None
            )
        
        # Verdict with enhanced styling
        verdict_color = "#FF6B6B" if report['overall_verdict'] == "LIKELY FAKE" else "#4ECDC4"
        verdict_icon = "⚠️" if report['overall_verdict'] == "LIKELY FAKE" else "✅"
        
        st.markdown(f"""
        <div style="background: {verdict_color}; padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin: 0; font-size: 2.5rem; font-weight: bold;">
                {verdict_icon} Overall Verdict: {report['overall_verdict']}
            </h2>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                Confidence: {float(report['confidence_score']):.1%}
            </p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Enhanced page config
    st.set_page_config(
        page_title="AI Deepfake Detector",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for modern styling
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: bold;
    }
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.3);
    }
    .css-1d391kg {
        background-color: #f8f9fa;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Modern header
    st.markdown('<h1 class="main-header">🎬 Frame-Based Deepfake Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Advanced Neural Network Powered Video Frame Analysis</p>', unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🤖 AI Models**\n\nMultiple trained neural networks")
    with col2:
        st.info("**🎬 Frame Analysis**\n\nDirect frame-level detection")
    with col3:
        st.info("**📊 Advanced Visualizations**\n\nComprehensive analysis charts")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
            <h3 style="color: white; text-align: center; margin: 0;">🔧 Configuration Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Model selection
        st.markdown("### 🤖 Model Selection")
        
        available_models = {}
        for model_name, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                available_models[model_name] = model_path
        
        if available_models:
            selected_model_name = st.selectbox(
                "Choose DF Model",
                options=list(available_models.keys()),
                help="Select a pre-trained deepfake detection model"
            )
            
            selected_model_path = available_models[selected_model_name]
            
            st.success(f"✅ **{selected_model_name}**")
            st.code(f"Path: {selected_model_path}", language="text")
            
        else:
            st.error("❌ No model files found!")
            selected_model_path = None
        
        # Analysis parameters
        st.markdown("### ⚙️ Analysis Parameters")
        
        frames_per_second = st.slider(
            "Frames per Second",
            min_value=1,
            max_value=10,
            value=2,
            help="Higher values = more detailed analysis"
        )
        
        # System info
        st.markdown("### 📊 System Info")
        st.info(f"**Device:** {'🚀 GPU' if torch.cuda.is_available() else '💻 CPU'}")
        st.info(f"**Available Models:** {len(available_models)}")
    
    # File upload section
    st.markdown("---")
    st.markdown("### 📁 Video Upload & Analysis")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your video file here",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None and selected_model_path:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            video_path = tmp_file.name
        
        # Video preview with modern layout
        col1, col2 = st.columns([2, 1])
        with col1:
            st.video(uploaded_file)
        with col2:
            st.markdown("""
            <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 1.5rem; border-radius: 10px;">
                <h4 style="color: white; margin: 0 0 1rem 0;">📹 Video Details</h4>
                <p style="color: white; margin: 0.5rem 0;"><strong>File:</strong> {}</p>
                <p style="color: white; margin: 0.5rem 0;"><strong>Size:</strong> {:.2f} MB</p>
                <p style="color: white; margin: 0.5rem 0;"><strong>Model:</strong> {}</p>
            </div>
            """.format(
                uploaded_file.name,
                uploaded_file.size / (1024*1024),
                selected_model_name
            ), unsafe_allow_html=True)
        
        # Analysis button
        if st.button("🚀 Start Deepfake Analysis", type="primary", use_container_width=True):
            try:
                pipeline = DeepFakeAnalysisPipeline(selected_model_path)
                
                with st.spinner("🔄 Initializing deepfake analysis pipeline..."):
                    report = pipeline.analyze_video(video_path, frames_per_second)
                
                if report:
                    pipeline.print_summary(report)
                    
                    # Enhanced results section
                    st.markdown("---")
                    st.markdown("### 📋 Detailed Analysis Results")
                    
                    with st.expander("📊 Advanced Statistics", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Processing Time", f"{len(report['detailed_analysis'])} frames")
                        with col2:
                            st.metric("Average Fake Confidence", f"{float(report['average_fake_confidence']):.3f}")
                        with col3:
                            st.metric("Average Real Confidence", f"{float(report['average_real_confidence']):.3f}")
                    
                    # Download section
                    st.markdown("### 💾 Export Results")
                    try:
                        json_report = json.dumps(report, indent=2, cls=NumpyEncoder)
                        st.download_button(
                            label="📥 Download Full Analysis Report (JSON)",
                            data=json_report,
                            file_name=f"deepfake_analysis_{uploaded_file.name}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    except Exception as e:
                        st.error(f"❌ Error creating download file: {e}")
            
            except Exception as e:
                st.error(f"❌ Analysis Error: {str(e)}")
        
        # Clean up
        try:
            os.unlink(video_path)
        except:
            pass
    
    elif uploaded_file is not None:
        st.error("❌ Please select a valid deepfake model before starting analysis")
    
    else:
        # Welcome message
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 3rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
            <h2 style="color: #2c3e50; margin: 0 0 1rem 0;">🎬 Ready to Analyze</h2>
            <p style="color: #2c3e50; margin: 0; font-size: 1.1rem;">
                Upload a video file to start frame-based deepfake detection
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 3rem;">
        <p>🛡️ <strong>Frame-Based Deepfake Detection System</strong> | Advanced Video Frame Analysis Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()