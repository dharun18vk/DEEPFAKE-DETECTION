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
from collections import defaultdict

# Set matplotlib to avoid GUI warnings
plt.set_loglevel('warning')

# =============================================================================
# MODEL PATHS CONFIGURATION - Define your model paths here
# =============================================================================
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
    "NEW B4 model":"DEEPFAKE MODELS/best_B4_model.pth",
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
            
            fps =    cap.get(cv2.CAP_PROP_FPS)
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
                if frame_count % 50 == 0:
                    progress = min(float(frame_count / total_frames), 1.0)
                    progress_bar.progress(progress)
                    status_text.text(f"🔄 Processed {frame_count}/{total_frames} frames")
            
            cap.release()
            progress_bar.progress(1.0)
            status.update(label=f"✅ **Frame Extraction Complete** - Extracted {len(extracted_frames)} frames", state="complete")
        return extracted_frames

class DeepFakeDetector:
    def __init__(self, model_path, model_name):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_path = model_path
        self.model_name = model_name
        self.model = self.load_model(model_path)
        self.transform = self.get_transform()
    
    def load_model(self, model_path):
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
            
        except Exception as e:
            st.error(f"❌ Error loading model {self.model_name}: {e}")
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
        if self.model is None:
            return None
            
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
            return None
    
    def predict_multiple_frames(self, frame_paths):
        if self.model is None:
            return []
            
        results = []
        for frame_info in frame_paths:
            result = self.predict_frame(frame_info['path'])
            if result:
                # Add frame metadata to result
                result.update({
                    'frame_number': frame_info['frame_number'],
                    'timestamp': frame_info['timestamp'],
                    'model_name': self.model_name
                })
                results.append(result)
        
        return results

class MultiModelDeepFakeAnalysisPipeline:
    def __init__(self):
        self.frame_extractor = FrameExtractor()
        self.results_dir = "analysis_results"
        os.makedirs(self.results_dir, exist_ok=True)
    
    def analyze_video_with_all_models(self, video_path, frames_per_second=1):
        # Modern header with gradient
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin-bottom: 2rem;">
            <h1 style="color: white; text-align: center; margin: 0; font-size: 2.5rem;">🚀 Multi-Model Deepfake Analysis</h1>
            <p style="color: white; text-align: center; margin: 0.5rem 0 0 0; font-size: 1.2rem;">Ensemble Detection with Multiple AI Models</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Step 1: Extract frames
        with st.container():
            st.markdown("### 🎬 STEP 1: Frame Extraction")
            frames = self.frame_extractor.extract_frames(video_path, frames_per_second)
            if not frames:
                st.error("❌ No frames extracted. Exiting.")
                return None
        
        # Step 2: Initialize all available models
        with st.container():
            st.markdown("### 🤖 STEP 2: Loading Deepfake Models")
            available_models = {}
            for model_name, model_path in MODEL_PATHS.items():
                if os.path.exists(model_path):
                    available_models[model_name] = model_path
            
            if not available_models:
                st.error("❌ No model files found!")
                return None
            
            # Display model loading status
            model_status = st.status("🔄 **Loading models...**", expanded=True)
            detectors = {}
            
            for model_name, model_path in available_models.items():
                with model_status:
                    st.write(f"📦 Loading {model_name}...")
                    detector = DeepFakeDetector(model_path, model_name)
                    if detector.model is not None:
                        detectors[model_name] = detector
                        st.success(f"✅ {model_name} loaded successfully")
                    else:
                        st.error(f"❌ Failed to load {model_name}")
            
            model_status.update(label=f"✅ **Models Loaded** - {len(detectors)} models ready", state="complete")
        
        # Step 3: Analyze with all models
        with st.container():
            st.markdown("### 🔍 STEP 3: Multi-Model Analysis")
            all_predictions = {}
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, (model_name, detector) in enumerate(detectors.items()):
                status_text.text(f"🧪 Analyzing with {model_name}...")
                predictions = detector.predict_multiple_frames(frames)
                all_predictions[model_name] = predictions
                
                progress = float((i + 1) / len(detectors))
                progress_bar.progress(progress)
            
            status_text.text("✅ All models completed analysis")
        
        # Step 4: Generate comprehensive report
        with st.container():
            st.markdown("### 📊 STEP 4: Generating Multi-Model Report")
            analysis_report = self.generate_multi_model_report(all_predictions, video_path)
        
        # Step 5: Visualize results
        with st.container():
            st.markdown("### 📈 STEP 5: Comprehensive Visualizations")
            self.create_multi_model_visualizations(all_predictions, analysis_report)
        
        return analysis_report
    
    def generate_multi_model_report(self, all_predictions, video_path):
        model_reports = {}
        ensemble_predictions = []
        
        for model_name, predictions in all_predictions.items():
            if not predictions:
                continue
                
            analyzed_frames = [pred for pred in predictions if 'prediction' in pred]
            total_frames = len(analyzed_frames)
            
            if total_frames == 0:
                continue
            
            fake_frames = sum(1 for pred in analyzed_frames if pred['prediction'] == 'FAKE')
            real_frames = total_frames - fake_frames
            
            fake_confidence_avg = float(np.mean([pred['confidence'] for pred in analyzed_frames if pred['prediction'] == 'FAKE'])) if fake_frames > 0 else 0.0
            real_confidence_avg = float(np.mean([pred['confidence'] for pred in analyzed_frames if pred['prediction'] == 'REAL'])) if real_frames > 0 else 0.0
            
            model_reports[model_name] = {
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
            
            # Collect predictions for ensemble analysis
            ensemble_predictions.extend(analyzed_frames)
        
        # Calculate ensemble results
        if ensemble_predictions:
            # Group by frame and take majority vote
            frame_predictions = defaultdict(list)
            for pred in ensemble_predictions:
                frame_key = pred['frame_number']
                frame_predictions[frame_key].append(pred['prediction'])
            
            ensemble_fake_frames = 0
            ensemble_real_frames = 0
            
            for frame_num, predictions in frame_predictions.items():
                fake_votes = predictions.count('FAKE')
                real_votes = predictions.count('REAL')
                if fake_votes > real_votes:
                    ensemble_fake_frames += 1
                else:
                    ensemble_real_frames += 1
            
            total_ensemble_frames = len(frame_predictions)
            ensemble_fake_percentage = (ensemble_fake_frames / total_ensemble_frames * 100) if total_ensemble_frames > 0 else 0
            
            model_reports['ENSEMBLE'] = {
                'total_frames_analyzed': int(total_ensemble_frames),
                'fake_frames_detected': int(ensemble_fake_frames),
                'real_frames_detected': int(ensemble_real_frames),
                'fake_percentage': float(ensemble_fake_percentage),
                'overall_verdict': "LIKELY FAKE" if ensemble_fake_frames > ensemble_real_frames else "LIKELY REAL",
                'confidence_score': float(max(
                    np.mean([r['average_fake_confidence'] for r in model_reports.values() if 'average_fake_confidence' in r]),
                    np.mean([r['average_real_confidence'] for r in model_reports.values() if 'average_real_confidence' in r])
                )) if model_reports else 0.0
            }
        
        report = {
            'video_path': video_path,
            'analysis_timestamp': str(np.datetime64('now')),
            'models_used': list(all_predictions.keys()),
            'model_reports': model_reports,
            'ensemble_report': model_reports.get('ENSEMBLE', {}),
            'all_predictions': all_predictions
        }
        
        report_path = os.path.join(self.results_dir, f"multi_model_analysis_{os.path.basename(video_path)}.json")
        try:
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            st.success(f"✅ Multi-model analysis report saved: {report_path}")
        except Exception as e:
            st.error(f"❌ Error saving report: {e}")
            return None
        
        return report
    
    def create_multi_model_visualizations(self, all_predictions, report):
        if not all_predictions:
            st.warning("⚠️ No prediction data for visualization")
            return
        
        # 1. Model Comparison Bar Chart
        st.markdown("### 📊 Model Comparison")
        
        model_names = []
        fake_percentages = []
        confidence_scores = []
        
        for model_name, model_report in report['model_reports'].items():
            if model_name != 'ENSEMBLE':
                model_names.append(model_name)
                fake_percentages.append(model_report['fake_percentage'])
                confidence_scores.append(model_report['confidence_score'])
        
        # Create comparison dataframe
        comparison_data = pd.DataFrame({
            'Model': model_names,
            'Fake Percentage': fake_percentages,
            'Confidence Score': confidence_scores
        })
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_fake_pct = px.bar(
                comparison_data,
                x='Model',
                y='Fake Percentage',
                title='Fake Percentage by Model',
                color='Fake Percentage',
                color_continuous_scale='RdYlGn_r'
            )
            fig_fake_pct.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_fake_pct, use_container_width=True)
        
        with col2:
            fig_conf = px.bar(
                comparison_data,
                x='Model',
                y='Confidence Score',
                title='Confidence Scores by Model',
                color='Confidence Score',
                color_continuous_scale='Blues'
            )
            fig_conf.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_conf, use_container_width=True)
        
        # 2. Model Agreement Heatmap
        st.markdown("### 🔥 Model Agreement Analysis")
        
        if len(model_names) > 1:
            # Calculate agreement matrix
            agreement_matrix = np.zeros((len(model_names), len(model_names)))
            frame_predictions = defaultdict(dict)
            
            for i, model_name in enumerate(model_names):
                for pred in all_predictions[model_name]:
                    frame_num = pred['frame_number']
                    frame_predictions[frame_num][model_name] = pred['prediction']
            
            for i, model1 in enumerate(model_names):
                for j, model2 in enumerate(model_names):
                    if i == j:
                        agreement_matrix[i][j] = 1.0
                    else:
                        agreements = 0
                        total_comparisons = 0
                        for frame_num, predictions in frame_predictions.items():
                            if model1 in predictions and model2 in predictions:
                                if predictions[model1] == predictions[model2]:
                                    agreements += 1
                                total_comparisons += 1
                        
                        if total_comparisons > 0:
                            agreement_matrix[i][j] = agreements / total_comparisons
                        else:
                            agreement_matrix[i][j] = 0
            
            fig_heatmap = px.imshow(
                agreement_matrix,
                x=model_names,
                y=model_names,
                title='Model Prediction Agreement Matrix',
                color_continuous_scale='Blues',
                aspect="auto"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 3. Detailed Model Results
        st.markdown("### 📈 Detailed Model Performance")
        
        for model_name, model_report in report['model_reports'].items():
            if model_name == 'ENSEMBLE':
                continue
                
            with st.expander(f"🔍 {model_name} Detailed Results", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Frames", model_report['total_frames_analyzed'])
                with col2:
                    st.metric("Real Frames", model_report['real_frames_detected'])
                with col3:
                    st.metric("Fake Frames", model_report['fake_frames_detected'])
                with col4:
                    st.metric("Fake %", f"{model_report['fake_percentage']:.1f}%")
                
                # Model-specific timeline
                model_predictions = all_predictions[model_name]
                if model_predictions:
                    df_model = pd.DataFrame(model_predictions)
                    fig_timeline = px.scatter(
                        df_model,
                        x='timestamp',
                        y='fake_probability',
                        color='prediction',
                        title=f'{model_name} - Fake Probability Timeline',
                        color_discrete_map={'REAL': '#4ECDC4', 'FAKE': '#FF6B6B'},
                        size='confidence',
                        hover_data=['frame_number']
                    )
                    st.plotly_chart(fig_timeline, use_container_width=True)
    
    def print_multi_model_summary(self, report):
        if report is None:
            st.error("❌ No report available to display")
            return
            
        # Enhanced summary with modern cards
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin: 2rem 0;">
            <h2 style="color: white; text-align: center; margin: 0; font-size: 2rem;">🎯 MULTI-MODEL DEEPFAKE ANALYSIS SUMMARY</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Main metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "📹 Video File",
                os.path.basename(report['video_path'])[:15] + "..." if len(os.path.basename(report['video_path'])) > 15 else os.path.basename(report['video_path'])
            )
        
        with col2:
            st.metric("🤖 Models Used", len(report['models_used']))
        
        with col3:
            st.metric("🎬 Total Frames", report['ensemble_report']['total_frames_analyzed'])
        
        with col4:
            st.metric("🔄 Analysis Time", report['analysis_timestamp'][11:19])
        
        # Model-specific results in a table
        st.markdown("### 📋 Model Results Summary")
        
        summary_data = []
        for model_name, model_report in report['model_reports'].items():
            if model_name != 'ENSEMBLE':
                summary_data.append({
                    'Model': model_name,
                    'Real Frames': model_report['real_frames_detected'],
                    'Fake Frames': model_report['fake_frames_detected'],
                    'Fake %': f"{model_report['fake_percentage']:.1f}%",
                    'Confidence': f"{model_report['confidence_score']:.3f}",
                    'Verdict': model_report['overall_verdict']
                })
        
        # Add ensemble result
        if 'ENSEMBLE' in report['model_reports']:
            ensemble_report = report['model_reports']['ENSEMBLE']
            summary_data.append({
                'Model': '🧠 ENSEMBLE',
                'Real Frames': ensemble_report['real_frames_detected'],
                'Fake Frames': ensemble_report['fake_frames_detected'],
                'Fake %': f"{ensemble_report['fake_percentage']:.1f}%",
                'Confidence': f"{ensemble_report['confidence_score']:.3f}",
                'Verdict': ensemble_report['overall_verdict']
            })
        
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Final verdict with enhanced styling
        ensemble_verdict = report['ensemble_report']['overall_verdict']
        verdict_color = "#FF6B6B" if ensemble_verdict == "LIKELY FAKE" else "#4ECDC4"
        verdict_icon = "⚠️" if ensemble_verdict == "LIKELY FAKE" else "✅"
        
        st.markdown(f"""
        <div style="background: {verdict_color}; padding: 2rem; border-radius: 15px; margin: 2rem 0; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: white; margin: 0; font-size: 2.5rem; font-weight: bold;">
                {verdict_icon} Ensemble Verdict: {ensemble_verdict}
            </h2>
            <p style="color: white; margin: 0.5rem 0 0 0; font-size: 1.2rem;">
                Based on {len(report['models_used'])} models | Fake Percentage: {report['ensemble_report']['fake_percentage']:.1f}%
            </p>
        </div>
        """, unsafe_allow_html=True)

def main():
    # Enhanced page config
    st.set_page_config(
        page_title="Multi-Model Deepfake Detector",
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
    st.markdown('<h1 class="main-header">🧠 Multi-Model Deepfake Detection System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #666; font-size: 1.2rem; margin-bottom: 2rem;">Ensemble AI Analysis with Multiple Neural Networks</p>', unsafe_allow_html=True)
    
    # Feature highlights
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**🤖 Multiple Models**\n\nEnsemble of trained neural networks")
    with col2:
        st.info("**🎯 Consensus Analysis**\n\nMajority voting for accurate results")
    with col3:
        st.info("**📊 Model Comparison**\n\nDetailed performance visualization")
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem; border-radius: 10px; margin-bottom: 2rem;">
            <h3 style="color: white; text-align: center; margin: 0;">🔧 Configuration Panel</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Show available models
        st.markdown("### 🤖 Available Models")
        
        available_models = {}
        for model_name, model_path in MODEL_PATHS.items():
            if os.path.exists(model_path):
                available_models[model_name] = model_path
        
        if available_models:
            st.success(f"✅ **{len(available_models)} models available**")
            for model_name in available_models.keys():
                st.write(f"• {model_name}")
        else:
            st.error("❌ No model files found!")
        
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
    st.markdown("### 📁 Video Upload & Multi-Model Analysis")
    
    uploaded_file = st.file_uploader(
        "Drag and drop your video file here",
        type=['mp4', 'avi', 'mov', 'mkv'],
        help="Supported formats: MP4, AVI, MOV, MKV"
    )
    
    if uploaded_file is not None:
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
                <p style="color: white; margin: 0.5rem 0;"><strong>Models:</strong> {}</p>
            </div>
            """.format(
                uploaded_file.name,
                uploaded_file.size / (1024*1024),
                len(available_models)
            ), unsafe_allow_html=True)
        
        # Analysis button
        if st.button("🚀 Start Multi-Model Deepfake Analysis", type="primary", use_container_width=True):
            try:
                pipeline = MultiModelDeepFakeAnalysisPipeline()
                
                with st.spinner("🔄 Initializing multi-model analysis pipeline..."):
                    report = pipeline.analyze_video_with_all_models(video_path, frames_per_second)
                
                if report:
                    pipeline.print_multi_model_summary(report)
                    
                    # Enhanced results section
                    st.markdown("---")
                    st.markdown("### 📋 Detailed Multi-Model Results")
                    
                    # Download section
                    st.markdown("### 💾 Export Results")
                    try:
                        json_report = json.dumps(report, indent=2, cls=NumpyEncoder)
                        st.download_button(
                            label="📥 Download Multi-Model Analysis Report (JSON)",
                            data=json_report,
                            file_name=f"multi_model_deepfake_analysis_{uploaded_file.name}.json",
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
    
    else:
        # Welcome message
        st.markdown("""
        <div style="background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%); padding: 3rem; border-radius: 15px; text-align: center; margin: 2rem 0;">
            <h2 style="color: #2c3e50; margin: 0 0 1rem 0;">🎬 Ready for Multi-Model Analysis</h2>
            <p style="color: #2c3e50; margin: 0; font-size: 1.1rem;">
                Upload a video file to start ensemble deepfake detection with all available models
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 3rem;">
        <p>🛡️ <strong>Multi-Model Deepfake Detection System</strong> | Ensemble AI Analysis Technology</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()