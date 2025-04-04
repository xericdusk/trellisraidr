import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import time
from datetime import datetime
import openai
import speech_recognition as sr
from scipy import signal
import base64
from io import BytesIO
import io
import re
import requests
import tempfile
import cv2  # For image processing
import gdown  # For downloading files from Google Drive

# Set page config
st.set_page_config(
    page_title="RAIDR - Tactical SIGINT Analyst",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure OpenAI API
try:
    openai.api_key = st.secrets["OPENAI_API_KEY"]
except KeyError:
    st.error("OpenAI API key is missing. Please set the OPENAI_API_KEY in Streamlit Cloud secrets.")
    st.stop()

# Load comprehensive signal database
@st.cache_data
def load_signal_database():
    return {
        "commercial_drone": {"freq_ranges": [[2.4e9, 2.5e9], [5.7e9, 5.9e9]], "bandwidths": [1e6, 4e6, 8e6, 20e6], "threat_level": "medium"},
        "wifi": {"freq_ranges": [[2.4e9, 2.5e9], [5.1e9, 5.9e9], [5.925e9, 7.125e9]], "bandwidths": [20e6, 40e6, 80e6, 160e6], "threat_level": "low"},
        "bluetooth": {"freq_ranges": [[2.4e9, 2.4835e9]], "bandwidths": [1e6], "threat_level": "low"},
        "lte_cellular": {"freq_ranges": [[700e6, 960e6], [1710e6, 2200e6], [2500e6, 2690e6]], "bandwidths": [1.4e6, 3e6, 5e6, 10e6, 15e6, 20e6], "threat_level": "low"},
        "5g_nr": {"freq_ranges": [[600e6, 6e9], [24e9, 40e9]], "bandwidths": [5e6, 10e6, 20e6, 50e6, 100e6, 200e6, 400e6], "threat_level": "low"},
        "pmr_radio": {"freq_ranges": [[440e6, 470e6]], "bandwidths": [12.5e3, 25e3], "threat_level": "low"},
        "military_vhf_uhf": {"freq_ranges": [[30e6, 88e6], [225e6, 400e6]], "bandwidths": [25e3, 5e6, 10e6], "threat_level": "high"},
        "fm_radio": {"freq_ranges": [[88e6, 108e6]], "bandwidths": [200e3], "threat_level": "low"},
        "am_radio": {"freq_ranges": [[535e3, 1605e3]], "bandwidths": [10e3], "threat_level": "low"},
        "amateur_radio": {"freq_ranges": [[1.8e6, 30e6], [50e6, 54e6], [144e6, 148e6], [420e6, 450e6]], "bandwidths": [10e3, 20e3, 100e3], "threat_level": "low"},
        "tv_broadcast": {"freq_ranges": [[54e6, 88e6], [174e6, 216e6], [470e6, 698e6]], "bandwidths": [6e6], "threat_level": "low"},
        "gps": {"freq_ranges": [[1575.42e6, 1575.42e6], [1227.6e6, 1227.6e6]], "bandwidths": [2e6], "threat_level": "low"},
        "emergency_services": {"freq_ranges": [[150e6, 174e6], [450e6, 470e6]], "bandwidths": [12.5e3, 25e3], "threat_level": "medium"},
        "radar": {"freq_ranges": [[2.9e9, 3.3e9], [8.5e9, 9.6e9], [33e9, 36e9]], "bandwidths": [10e6, 50e6, 100e6], "threat_level": "high"},
        "air_traffic_control": {"freq_ranges": [[108e6, 137e6]], "bandwidths": [25e3], "threat_level": "low"},
        "satcom": {"freq_ranges": [[1.6e9, 1.7e9], [3.6e9, 4.2e9], [7.25e9, 7.75e9], [10.7e9, 12.75e9]], "bandwidths": [1e6, 36e6], "threat_level": "high"}
    }

signal_db = load_signal_database()

# NATO phonetic alphabet for numbers
nato_numbers = {
    '0': 'Zero', '1': 'One', '2': 'Two', '3': 'Three', '4': 'Four',
    '5': 'Five', '6': 'Six', '7': 'Seven', '8': 'Eight', '9': 'Niner'
}

def format_frequency(freq_mhz):
    freq_str = f"{freq_mhz:.3f}"
    result = ""
    for char in freq_str:
        if char in nato_numbers:
            result += f"{nato_numbers[char]} "
        elif char == '.':
            result += "Point "
    return result.strip() + " Megahertz"

def format_number(num):
    num_str = str(num)
    return " ".join(nato_numbers[char] for char in num_str if char in nato_numbers)

def download_file_from_google_drive(file_url):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.tmp') as temp_file:
            temp_path = temp_file.name
            gdown.download(file_url, temp_path, quiet=False)
        
        with open(temp_path, 'rb') as f:
            raw_bytes = f.read()
        
        os.unlink(temp_path)
        
        if raw_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            file_type = "png"
        elif raw_bytes.startswith(b'{') or raw_bytes.startswith(b'['):
            file_type = "json"
        elif b'\n' in raw_bytes and (b'{' in raw_bytes or b'[' in raw_bytes):
            file_type = "jsonl"
        elif b',' in raw_bytes and b'\n' in raw_bytes:
            file_type = "csv"
        else:
            st.error("Downloaded file is not a recognized format.")
            return None, None, None
        
        st.info(f"Downloaded file size: {len(raw_bytes)} bytes, detected as {file_type}")
        
        file_name = f"downloaded_file.{file_type}"
        file_content = io.BytesIO(raw_bytes)
        return file_content, file_name, raw_bytes
    except Exception as e:
        st.error(f"Error downloading file from Google Drive: {str(e)}")
        return None, None, None

def check_file_header(file, header_bytes):
    try:
        file.seek(0)
        file_start = file.read(len(header_bytes))
        file.seek(0)
        return file_start == header_bytes
    except Exception as e:
        print(f"Error checking file header: {str(e)}")
        return False

def process_json_array_signals(json_data):
    try:
        detected_signals = []
        if isinstance(json_data, list):
            timestamp = "Unknown"
            if len(json_data) > 0 and "time" in json_data[0]:
                timestamp = json_data[0]["time"]
            
            min_freq = float('inf')
            max_freq = float('-inf')
            
            for signal in json_data:
                freq = signal.get("centerFreq(Hz)", signal.get("frequency", signal.get("freq", 0)))
                bandwidth = signal.get("bandwidth(Hz)", signal.get("bandwidth", 1e6))
                power = signal.get("peakLevel(dBm)", signal.get("power_dbm", signal.get("power", -100)))
                
                if freq < min_freq:
                    min_freq = freq
                if freq > max_freq:
                    max_freq = freq
                
                signal_type = classify_signal(freq, bandwidth)
                
                detected_signals.append({
                    "frequency": float(freq),
                    "power_dbm": float(power),
                    "bandwidth": float(bandwidth),
                    "type": signal_type["type"],
                    "threat_level": signal_type["threat_level"],
                    "confidence": signal_type["confidence"]
                })
            
            if min_freq == float('inf'):
                min_freq = 0
            if max_freq == float('-inf'):
                max_freq = 0
                
            return {
                "timestamp": timestamp,
                "frequency_range": [min_freq, max_freq],
                "detected_signals": detected_signals,
                "is_signal_array": True
            }
        return {}
    except Exception as e:
        st.error(f"Error processing JSON array signals: {str(e)}")
        return {}

def process_csv_file(csv_file):
    try:
        df = pd.read_csv(csv_file)
        required_cols = ['frequency', 'power_dbm']
        if not all(col in df.columns for col in required_cols):
            if 'freq' in df.columns:
                df.rename(columns={'freq': 'frequency'}, inplace=True)
            if 'power' in df.columns:
                df.rename(columns={'power': 'power_dbm'}, inplace=True)
            
            if not all(col in df.columns for col in required_cols):
                st.warning(f"CSV file missing required columns: {required_cols}. Attempting to infer data structure.")
                if len(df.columns) >= 2:
                    df.rename(columns={df.columns[0]: 'frequency', df.columns[1]: 'power_dbm'}, inplace=True)
        
        if df['frequency'].dtype == object:
            df['frequency'] = df['frequency'].str.replace('MHz', '').str.replace('GHz', '000').str.strip().astype(float)
        if df['power_dbm'].dtype == object:
            df['power_dbm'] = df['power_dbm'].str.replace('dBm', '').str.strip().astype(float)
        
        if 'bandwidth' not in df.columns:
            df['bandwidth'] = 1e6
        
        signals = []
        for _, row in df.iterrows():
            freq = float(row['frequency'])
            power = float(row['power_dbm'])
            bw = float(row['bandwidth']) if 'bandwidth' in row and pd.notna(row['bandwidth']) else 1e6
            signal_type = classify_signal(freq, bw)
            signals.append({
                "frequency": freq,
                "power_dbm": power,
                "bandwidth": bw,
                "type": signal_type["type"],
                "threat_level": signal_type["threat_level"],
                "confidence": signal_type["confidence"]
            })
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        min_freq = df['frequency'].min()
        max_freq = df['frequency'].max()
        
        return {
            "timestamp": timestamp,
            "frequency_range": [min_freq, max_freq],
            "detected_signals": signals,
            "is_csv": True
        }
    except Exception as e:
        st.error(f"Error processing CSV file: {str(e)}")
        return {}

def process_waterfall_image(image_data):
    try:
        file_bytes = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not load image")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        timestamp = "2025-03-31 18:46:49 to 18:51:06"
        freq_range = [10e6, 6000e6]
        power_range = [0.46, 20.69]

        height, width = gray.shape
        plot_top = int(height * 0.1)
        plot_bottom = int(height * 0.9)
        plot_left = int(width * 0.05)
        plot_right = int(width * 0.95)
        plot_img = gray[plot_top:plot_bottom, plot_left:plot_right]

        _, thresh = cv2.threshold(plot_img, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detected_signals = []
        plot_width = plot_right - plot_left
        freq_start, freq_end = freq_range

        def x_to_freq(x):
            norm_x = x / plot_width
            log_freq = np.log10(freq_start) + norm_x * (np.log10(freq_end) - np.log10(freq_start))
            return 10 ** log_freq

        def intensity_to_power(intensity):
            norm_intensity = intensity / 255.0
            return power_range[0] + norm_intensity * (power_range[1] - power_range[0])

        for contour in contours:
            M = cv2.moments(contour)
            if M["m00"] == 0:
                continue
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            freq = x_to_freq(cx)
            intensity = plot_img[cy, cx]
            power_dbm = intensity_to_power(intensity)
            x, y, w, h = cv2.boundingRect(contour)
            bw = (w / plot_width) * (freq_end - freq_start)

            signal_type = classify_signal(freq, bw)
            detected_signals.append({
                "frequency": float(freq),
                "power_dbm": float(power_dbm),
                "bandwidth": float(bw),
                "type": signal_type["type"],
                "threat_level": signal_type["threat_level"],
                "confidence": signal_type["confidence"]
            })

        detected_signals.sort(key=lambda x: x["frequency"])
        return {
            "timestamp": timestamp,
            "frequency_range": freq_range,
            "detected_signals": detected_signals,
            "is_waterfall": True,
            "image_data": image_data
        }
    except Exception as e:
        st.error(f"Error processing waterfall image: {str(e)}")
        return {}

def load_scan_file(uploaded_file):
    try:
        result = {
            "timestamp": "Unknown time",
            "frequency_range": [0, 0],
            "detected_signals": [],
            "is_waterfall": False,
            "is_csv": False,
            "is_signal_array": False
        }
        
        file_extension = ""
        if hasattr(uploaded_file, 'name'):
            file_extension = uploaded_file.name.split('.')[-1].lower()
            st.info(f"Processing file with extension: {file_extension}")
        
        if file_extension == 'png' or check_file_header(uploaded_file, b'\x89PNG\r\n\x1a\n'):
            result.update(process_waterfall_image(uploaded_file))
            return result
        
        elif file_extension == 'csv':
            result.update(process_csv_file(uploaded_file))
            return result
        
        elif file_extension == 'jsonl':
            content = uploaded_file.read().decode('utf-8')
            for line in content.split('\n'):
                if line.strip():
                    data = json.loads(line.strip())
                    result.update(data)
                    return result
            return result
        
        elif file_extension == 'json':
            content = uploaded_file.read().decode('utf-8')
            data = json.loads(content)
            
            if isinstance(data, list) and len(data) > 0 and "centerFreq(Hz)" in data[0]:
                signal_data = process_json_array_signals(data)
                if signal_data:
                    result.update(signal_data)
                    return result
            
            if "timestamp_start" in data and "timestamp_end" in data:
                result.update({
                    "timestamp": f"{data['timestamp_start']} to {data['timestamp_end']}",
                    "frequency_range": data.get("frequency_range", [0, 0]),
                    "detected_signals": data.get("detected_signals", []),
                    "is_waterfall": True
                })
                return result
            
            if "metadata" in data and "top_signals" in data:
                timestamp = data["top_signals"][0]["timestamp"] if data["top_signals"] else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                freq_range = [data["metadata"]["frequency_range_mhz"][0] * 1e6, data["metadata"]["frequency_range_mhz"][1] * 1e6]
                detected_signals = []
                for signal in data["top_signals"]:
                    freq_hz = signal["frequency_mhz"] * 1e6
                    power_db = signal["power_db"]
                    bandwidth = data["metadata"]["bin_width_hz"]
                    signal_type = classify_signal(freq_hz, bandwidth)
                    detected_signals.append({
                        "frequency": float(freq_hz),
                        "power_dbm": float(power_db),
                        "bandwidth": float(bandwidth),
                        "type": signal_type["type"],
                        "threat_level": signal_type["threat_level"],
                        "confidence": signal_type["confidence"]
                    })
                result.update({
                    "timestamp": timestamp,
                    "frequency_range": freq_range,
                    "detected_signals": detected_signals,
                    "is_signal_array": True
                })
                return result
            
            result.update(data)
            return result
        
        else:
            try:
                uploaded_file.seek(0)
                content_start = uploaded_file.read(1024).decode('utf-8', errors='ignore')
                uploaded_file.seek(0)
                
                if content_start.strip().startswith('{') or content_start.strip().startswith('['):
                    try:
                        content = uploaded_file.read().decode('utf-8')
                        data = json.loads(content)
                        if "metadata" in data and "top_signals" in data:
                            timestamp = data["top_signals"][0]["timestamp"] if data["top_signals"] else datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            freq_range = [data["metadata"]["frequency_range_mhz"][0] * 1e6, data["metadata"]["frequency_range_mhz"][1] * 1e6]
                            detected_signals = []
                            for signal in data["top_signals"]:
                                freq_hz = signal["frequency_mhz"] * 1e6
                                power_db = signal["power_db"]
                                bandwidth = data["metadata"]["bin_width_hz"]
                                signal_type = classify_signal(freq_hz, bandwidth)
                                detected_signals.append({
                                    "frequency": float(freq_hz),
                                    "power_dbm": float(power_db),
                                    "bandwidth": float(bandwidth),
                                    "type": signal_type["type"],
                                    "threat_level": signal_type["threat_level"],
                                    "confidence": signal_type["confidence"]
                                })
                            result.update({
                                "timestamp": timestamp,
                                "frequency_range": freq_range,
                                "detected_signals": detected_signals,
                                "is_signal_array": True
                            })
                            return result
                        result.update(data)
                        return result
                    except Exception as e:
                        st.warning(f"Failed to parse as JSON: {str(e)}")
            except Exception as e:
                st.error(f"Error determining file type: {str(e)}")
                return result
                
    except json.JSONDecodeError as e:
        st.error(f"Invalid JSON format: {str(e)}")
        return result
    except Exception as e:
        st.error(f"Error loading scan file: {str(e)}")
        return result

def process_iq_data(iq_data, center_freq, sample_rate):
    freqs, psd = signal.welch(iq_data, fs=sample_rate, nperseg=1024, scaling='spectrum')
    freq_axis = freqs + (center_freq - sample_rate/2)
    peaks, _ = signal.find_peaks(psd, height=np.mean(psd)*3, distance=10)
    detected_signals = []
    for peak in peaks:
        peak_freq = freq_axis[peak]
        peak_power = 10 * np.log10(psd[peak])
        bw = estimate_bandwidth(psd, peak, sample_rate, len(psd))
        signal_type = classify_signal(peak_freq, bw)
        detected_signals.append({
            "frequency": float(peak_freq),
            "power_dbm": float(peak_power),
            "bandwidth": float(bw),
            "type": signal_type["type"],
            "threat_level": signal_type["threat_level"],
            "confidence": signal_type["confidence"]
        })
    return {"freq_axis": freq_axis.tolist(), "psd": psd.tolist(), "detected_signals": detected_signals}

def estimate_bandwidth(psd, peak_idx, sample_rate, n_points):
    peak_power = psd[peak_idx]
    threshold = peak_power / 2
    lower, upper = peak_idx, peak_idx
    while lower > 0 and psd[lower] > threshold:
        lower -= 1
    while upper < len(psd)-1 and psd[upper] > threshold:
        upper += 1
    return (upper - lower) * (sample_rate / n_points)

def classify_signal(frequency, bandwidth):
    best_match = {"type": "unknown", "threat_level": "unknown", "confidence": 0.0}
    for sig_type, sig_info in signal_db.items():
        in_range = any(freq_range[0] <= frequency <= freq_range[1] for freq_range in sig_info["freq_ranges"])
        if not in_range:
            continue
        bw_match = any(0.5 * bw <= bandwidth <= 2.0 * bw for bw in sig_info["bandwidths"])
        confidence = 0.7 if in_range else 0.0
        confidence += 0.3 if bw_match else 0.0
        if confidence > best_match["confidence"]:
            best_match = {"type": sig_type, "threat_level": sig_info["threat_level"], "confidence": confidence}
    if best_match["confidence"] == 0.0:
        if frequency < 30e6:
            best_match = {"type": "hf_signal", "threat_level": "low", "confidence": 0.3}
        elif frequency < 300e6:
            best_match = {"type": "vhf_signal", "threat_level": "low", "confidence": 0.3}
        elif frequency < 1e9:
            best_match = {"type": "uhf_signal", "threat_level": "low", "confidence": 0.3}
        else:
            best_match = {"type": "microwave_signal", "threat_level": "low", "confidence": 0.3}
    return best_match

def text_to_speech(text, voice="onyx"):
    try:
        lines = text.split('\n')
        processed_text = ""
        for line in lines:
            line = line.strip()
            if not line:
                processed_text += " Break. "
                continue
            
            freq_match = re.search(r'(\d+\.\d+|\d+)\s*MHz', line, re.IGNORECASE)
            if freq_match:
                freq = float(freq_match.group(1))
                processed_text += f"Frequency {format_frequency(freq)}. "
            else:
                words = line.split()
                for i, word in enumerate(words):
                    if re.match(r'^\d+$', word):
                        words[i] = format_number(word)
                    elif re.match(r'^\d+\.\d+$', word):
                        words[i] = " ".join(nato_numbers[c] if c in nato_numbers else "Point" if c == '.' else c for c in word)
                processed_text += " ".join(words) + ". "
        
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as temp_file:
            temp_path = temp_file.name
        
        headers = {
            "Authorization": f"Bearer {openai.api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": "tts-1",
            "input": processed_text,
            "voice": voice,
            "response_format": "mp3",
            "speed": 0.9
        }
        
        response = requests.post(
            "https://api.openai.com/v1/audio/speech",
            headers=headers,
            json=data,
            stream=True
        )
        
        if response.status_code == 200:
            with open(temp_path, 'wb') as f:
                for data_chunk in response.iter_content(chunk_size=1024):
                    if data_chunk:
                        f.write(data_chunk)
            
            with open(temp_path, 'rb') as f:
                audio_data = f.read()
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            os.unlink(temp_path)
            
            audio_html = f"""
            <audio controls autoplay>
                <source src="data:audio/mp3;base64,{audio_base64}" type="audio/mp3">
                Your browser does not support the audio element.
            </audio>
            """
            st.markdown(audio_html, unsafe_allow_html=True)
            st.success("Voice transmission complete")
            return True
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        st.error(f"Text-to-speech error: {str(e)}")
        return False

def speech_to_text():
    st.warning("Speech-to-text is not supported in the cloud environment due to microphone access limitations.")
    return ""

def plot_spectrum(scan_data):
    if scan_data.get("is_waterfall", False) or scan_data.get("is_csv", False) or scan_data.get("is_signal_array", False):
        fig, ax = plt.subplots(figsize=(10, 5))
        signals = scan_data.get("detected_signals", [])
        if not signals:
            st.warning("No signals to plot in data.")
            return None
        
        freqs = [signal["frequency"] / 1e6 for signal in signals]
        powers = [signal.get("power_dbm", -100) for signal in signals]
        ax.scatter(freqs, powers, c='red', label='Detected Signals')
        
        for signal in signals:
            freq = signal["frequency"] / 1e6
            power = signal.get("power_dbm", -100)
            label = signal.get("type", "unknown")
            color = 'green' if signal.get("threat_level") == "low" else 'orange' if signal.get("threat_level") == "medium" else 'red'
            ax.plot(freq, power, 'o', markersize=8, color=color)
            ax.annotate(label, (freq, power), xytext=(0, 10), textcoords='offset points', ha='center')
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power (dBm)')
        center_freq = scan_data.get("center_freq", 0) / 1e6
        ax.set_title(f'RF Spectrum at {center_freq} MHz')
        ax.grid(True, alpha=0.3)
        
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        img_str = base64.b64encode(buf.read()).decode('utf-8')
        plt.close(fig)
        return img_str
    else:
        st.warning("Unsupported scan data format for plotting.")
        return None

def get_available_scans(uploaded_files):
    scans = []
    for i, file in enumerate(uploaded_files):
        file.seek(0)
        scan_data = load_scan_file(file)
        if scan_data:
            file_type = "unknown"
            if hasattr(file, 'name'):
                ext = file.name.split('.')[-1].lower()
                file_type = ext
            else:
                file.seek(0)
                try:
                    content_start = file.read(8)
                    file.seek(0)
                    if content_start.startswith(b'\x89PNG\r\n\x1a\n'):
                        file_type = "png"
                    elif scan_data.get("is_csv", False):
                        file_type = "csv"
                    elif scan_data.get("is_waterfall", False):
                        file_type = "json"
                    elif scan_data.get("is_signal_array", False):
                        file_type = "json"
                except Exception:
                    file.seek(0)
            
            timestamp = scan_data.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            try:
                if scan_data.get("is_waterfall") or scan_data.get("is_csv", False) or scan_data.get("is_signal_array", False):
                    freq_range = scan_data.get("frequency_range", [0, 0])
                    center_freq = freq_range[0] / 1e6 if freq_range and len(freq_range) > 0 else 0
                else:
                    center_freq = scan_data.get("center_freq", 0) / 1e6
            except Exception:
                center_freq = 0
            
            scans.append({
                "id": i,
                "timestamp": timestamp,
                "center_freq": center_freq,
                "file_type": file_type,
                "scan_data": scan_data,
                "file_content": file
            })
    return scans

def prepare_llm_context(selected_scan_data, max_signals_per_scan=25):
    context = "SIGINT report follows. "
    for scan in selected_scan_data:
        scan_data = scan["scan_data"]
        context += f"Scan ID: {format_number(scan['id'])}. "
        context += f"Time: {scan['timestamp']}. "
        signals = scan_data.get("detected_signals", [])
        
        sorted_signals = sorted(
            signals, 
            key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(x.get("threat_level", "unknown"), 3)
        )
        
        limited_signals = sorted_signals[:max_signals_per_scan]
        
        if limited_signals:
            total_count = len(signals)
            showing_count = len(limited_signals)
            context += f"Signals detected: {showing_count} of {total_count} shown, prioritized by threat level. "
            for signal in limited_signals:
                freq_mhz = signal.get('frequency', 0) / 1e6
                context += f"Frequency {format_frequency(freq_mhz)}. "
                context += f"Type {signal.get('type', 'unknown')}. "
                context += f"Threat {signal.get('threat_level', 'unknown')}. Break. "
        else:
            context += "No signals detected. Break. "
    
    return context

def query_chatgpt(query, context):
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are RAIDR (pronounced 'Raider'), a tactical SIGINT analyst assistant. Respond in brief tactical radio format using NATO phonetic numbers and 'point' for decimals (e.g., 'One Two Five Point One Two Five' for 125.125 MHz). Do not start with 'This is RAIDR' or end with 'Over and Out'."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
            ],
            max_tokens=500,
            temperature=0.5
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error querying ChatGPT: {str(e)}")
        return "Unable to process query."

def query_advanced_analysis(query, context):
    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a SIGINT analyst."},
                {"role": "user", "content": f"Context (scan data):\n{context}\n\nQuery: {query}"}
            ],
            max_tokens=1500,
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error querying GPT-4 Turbo: {str(e)}")
        return "Unable to process query."

def find_best_trellis_frequency(scan_data):
    """Analyze scan data and recommend the best TrellisWare frequency using GPT-4 Turbo."""
    trellis_specs = """
    TrellisWare Radio Specifications:
    - Frequency Range:
      • L-UHF: 225–450 MHz
      • U-UHF: 698–970 MHz
      • L/S Bands: 1250–2600 MHz
    - Bandwidth/Channel Spacing:
      • Wideband: 1.2, 3.6, 10, 20 MHz
      • Narrowband: 6.25, 8.33, 12.5, 20, 25, 50 kHz
    - Transmit Power: .001 mW, .01 mW, .1 mW, 10 mW, 100 mW, 250 mW, 500 mW, 1W, 2W
    """

    # Prepare signal data context
    signals = scan_data.get("detected_signals", [])
    if not signals:
        return None, "No signals detected in the scan data."

    context = "Detected Signals:\n"
    for signal in signals:
        freq_mhz = signal.get("frequency", 0) / 1e6
        power_dbm = signal.get("power_dbm", -100)
        bandwidth_hz = signal.get("bandwidth", 1e6)
        context += f"- Frequency: {freq_mhz:.3f} MHz, Power: {power_dbm:.1f} dBm, Bandwidth: {bandwidth_hz/1e6:.3f} MHz\n"

    query = f"""
    Given the following TrellisWare radio specifications and detected signals, determine the best center frequency (in MHz) for the radio to operate with minimal interference. Return only a single number with three decimal places (e.g., 123.456), nothing else. If no suitable frequency is found, return 'N/A'.

    {trellis_specs}

    {context}
    """

    try:
        client = openai.OpenAI(api_key=openai.api_key)
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[
                {"role": "system", "content": "You are a SIGINT analyst tasked with finding the best frequency for a TrellisWare radio based on interference data. Provide only a single number with three decimal places (e.g., 123.456) or 'N/A' if no frequency is suitable, with no additional text."},
                {"role": "user", "content": query}
            ],
            max_tokens=10,  # Expecting a short numeric response or 'N/A'
            temperature=0.2  # Lower temperature for strict adherence
        )
        freq_response = response.choices[0].message.content.strip()
        
        # Validate the response
        if freq_response == "N/A":
            return None, "No suitable frequency found by AI."
        try:
            recommended_freq = float(freq_response)
            valid_ranges = [(225, 450), (698, 970), (1250, 2600)]
            if any(start <= recommended_freq <= end for start, end in valid_ranges):
                return recommended_freq, None
            else:
                return None, f"Recommended frequency {recommended_freq} MHz is outside TrellisWare radio ranges."
        except ValueError:
            return None, f"Invalid frequency response from AI: {freq_response}"
    except Exception as e:
        return None, f"Error querying GPT-4 Turbo: {str(e)}"

def main():
    st.markdown("""
    <style>
    .main {background-color: #1E1E1E; color: #DCDCDC;}
    .st-bw {background-color: #2D2D2D;}
    .stTextInput > div > div > input {background-color: #3D3D3D; color: #DCDCDC;}
    .tactical-header {color: #00FF00; text-align: center; font-family: 'Courier New', monospace; margin-bottom: 20px;}
    .tactical-text {font-family: 'Courier New', monospace; background-color: #2D2D2D; padding: 10px; border-radius: 5px; border-left: 3px solid #00FF00; white-space: pre-wrap;}
    .threat-high {color: #FF5555; font-weight: bold;}
    .threat-medium {color: #FFAA00; font-weight: bold;}
    .threat-low {color: #55FF55;}
    .analyze-btn {background-color: #444; color: #00FF00; border: none; padding: 5px 10px; cursor: pointer; border-radius: 4px; float: right; margin-top: 5px;}
    .analyze-btn:hover {background-color: #555;}
    .chat-container {max-width: 100%; margin: 0 auto;}
    .message {padding: 10px; margin-bottom: 10px; border-radius: 5px;}
    .user-message {background-color: #2D2D2D; border-left: 3px solid #00AAFF;}
    .ai-message {background-color: #333333; border-left: 3px solid #00FF00;}
    .lime-green {color: #00FF00; font-size: 48px; font-weight: bold; text-align: center;}
    </style>
    <script>
    function switchToTab(tabIndex) {
        window.parent.postMessage({
            type: 'streamlit:setComponentValue',
            value: tabIndex
        }, '*');
    }
    </script>
    """, unsafe_allow_html=True)
    
    st.markdown('<h1 class="tactical-header">RAIDR: Tactical SIGINT Analyst</h1>', unsafe_allow_html=True)
    
    if 'active_tab' not in st.session_state:
        st.session_state.active_tab = 0
    if 'advanced_analysis_data' not in st.session_state:
        st.session_state.advanced_analysis_data = {'tactical_response': '', 'additional_question': ''}
    if 'advanced_chat_history' not in st.session_state:
        st.session_state.advanced_chat_history = []
    
    with st.sidebar:
        st.header("Scan Controls")
        st.subheader("Load Scan from Google Drive Link")
        file_url = st.text_input("Enter Google Drive shareable link to a file:", key="file_url")
        st.subheader("Upload Local Scan File")
        uploaded_file = st.file_uploader("Choose a file", type=["png", "json", "jsonl", "csv"], key="file_uploader")
        
        model_options = ["gpt-3.5-turbo", "gpt-4-turbo"]
        selected_model = st.selectbox("Select AI Model", model_options, index=1)
        
        limit_signals = st.checkbox("Limit number of signals for analysis", value=True)
        if limit_signals:
            max_signals = st.slider("Max signals to include per scan", min_value=5, max_value=100, value=25, step=5)
        else:
            max_signals = 10000
            st.info("Including all signals. This may exceed token limits with large datasets.")
        
        uploaded_files = []
        if file_url:
            with st.spinner("Downloading file from Google Drive..."):
                file_content, file_name, raw_bytes = download_file_from_google_drive(file_url)
                if file_content and file_name:
                    uploaded_file_from_drive = type('UploadedFile', (), {
                        'read': lambda self: file_content.read(),
                        'seek': lambda self, pos: file_content.seek(pos),
                        'name': file_name
                    })()
                    file_content.seek(0)
                    uploaded_file_from_drive.seek(0)
                    uploaded_files.append(uploaded_file_from_drive)
                    st.success(f"Successfully downloaded file: {file_name}")
        
        if uploaded_file is not None:
            uploaded_files.append(uploaded_file)
            st.success(f"Successfully uploaded file: {uploaded_file.name}")
        
        if not uploaded_files:
            st.info("No files uploaded or downloaded. Please upload a file (PNG, JSON, JSONL, CSV) or provide a valid Google Drive shareable link.")
        
        if uploaded_files:
            scans = get_available_scans(uploaded_files)
            if not scans:
                st.info("No valid scans loaded. Please ensure the file is a valid format.")
            else:
                scan_options = {f"{scan['id']} - {scan['timestamp'][:16]} ({scan['center_freq']} MHz) [{scan['file_type']}]": scan for scan in scans}
                selected_scans = st.multiselect("Select scans for analysis", options=list(scan_options.keys()), default=list(scan_options.keys()))
                selected_scan_data = [scan_options[scan] for scan in selected_scans]
    
    tab1, tab2, tab3, tab4 = st.tabs(["Spectrum Analyzer", "Tactical SIGINT", "Advanced Analysis", "Trellis"])
    
    if st.session_state.active_tab == 0:
        active_tab = tab1
    elif st.session_state.active_tab == 1:
        active_tab = tab2
    elif st.session_state.active_tab == 2:
        active_tab = tab3
    else:
        active_tab = tab4
    
    with tab1:
        st.header("RF Spectrum Analysis")
        if 'selected_scan_data' in locals() and selected_scan_data:
            scan = selected_scan_data[0]
            scan_data = scan["scan_data"]
            file_type = scan["file_type"]
            
            if file_type == "png":
                st.image(scan["file_content"], use_container_width=True, caption="Original Waterfall Plot")
            else:
                img_str = plot_spectrum(scan_data)
                if img_str:
                    st.image(f"data:image/png;base64,{img_str}", use_container_width=True)
            
            st.subheader("Detected Signals")
            signals = scan_data.get("detected_signals", [])
            if signals:
                signals_df = pd.DataFrame([
                    {
                        "Frequency (MHz)": signal.get("frequency", 0) / 1e6,
                        "Power (dBm)": signal.get("power_dbm", 0),
                        "Bandwidth (kHz)": signal.get("bandwidth", 0) / 1e3 if "bandwidth" in signal else "N/A",
                        "Type": signal.get("type", "unknown"),
                        "Threat Level": signal.get("threat_level", "unknown"),
                        "Confidence": f"{signal.get('confidence', 0):.1f}"
                    } for signal in signals
                ])
                def highlight_threats(val):
                    return 'color: red; font-weight: bold' if val == "high" else 'color: orange; font-weight: bold' if val == "medium" else 'color: green' if val == "low" else ''
                styled_df = signals_df.style.applymap(highlight_threats, subset=["Threat Level"])
                st.dataframe(styled_df, use_container_width=True)
                
                st.subheader("Signal Statistics")
                col1, col2, col3 = st.columns(3)
                with col1:
                    threat_counts = signals_df["Threat Level"].value_counts()
                    st.metric("High Threat Signals", threat_counts.get("high", 0))
                with col2:
                    st.metric("Medium Threat Signals", threat_counts.get("medium", 0))
                with col3:
                    st.metric("Low Threat Signals", threat_counts.get("low", 0))
            else:
                st.info("No signals detected in this scan.")
        else:
            st.info("No scans available for analysis. Please upload a file (PNG, JSON, JSONL, CSV) or provide a valid Google Drive shareable link.")
    
    with tab2:
        st.header("RAIDR Tactical SIGINT Interface")
        if 'selected_scan_data' in locals() and selected_scan_data:
            context = prepare_llm_context(selected_scan_data, max_signals_per_scan=max_signals)
            
            if 'tactical_query' not in st.session_state:
                st.session_state.tactical_query = ""
            
            col_query, col_clear = st.columns([3, 1])
            with col_query:
                query = st.text_input("Enter tactical query:", value=st.session_state.tactical_query, key="text_query")
            with col_clear:
                if st.button("New Entry"):
                    st.session_state.tactical_query = ""
                    st.rerun()
            
            if query != st.session_state.tactical_query:
                st.session_state.tactical_query = query
            
            voice_options = ["alloy", "echo", "fable", "onyx", "nova", "shimmer"]
            selected_voice = st.selectbox("Select TTS Voice", voice_options, index=3)
            
            col1, col2 = st.columns([1, 1])
            with col1:
                if st.button("Send Query"):
                    if query:
                        with st.spinner(f"RAIDR processing with {selected_model}..."):
                            response = query_chatgpt(query, context)
                            st.session_state.last_response = response
                            st.session_state.show_response = True
            with col2:
                if st.button("Voice Query"):
                    with st.spinner("Listening for voice query..."):
                        voice_query = speech_to_text()
                        if voice_query:
                            st.success(f"Voice query detected: \"{voice_query}\"")
                            with st.spinner(f"RAIDR processing with {selected_model}..."):
                                response = query_chatgpt(voice_query, context)
                                st.session_state.last_response = response
                                st.session_state.show_response = True
            
            if st.session_state.get('show_response', False):
                st.subheader("RAIDR Response:")
                response_container = st.container()
                with response_container:
                    st.markdown(f'<div class="tactical-text">{st.session_state.last_response}</div>', unsafe_allow_html=True)
                    if st.button("Send to Advanced Analysis"):
                        st.session_state.advanced_analysis_data['tactical_response'] = st.session_state.last_response
                        st.session_state.active_tab = 2
                        st.rerun()
                
                if st.button("Speak Response"):
                    text_to_speech(st.session_state.last_response, selected_voice)
                    
            with st.expander("Signal Summary"):
                signals_count = sum(len(scan["scan_data"].get("detected_signals", [])) for scan in selected_scan_data)
                included_count = min(signals_count, max_signals * len(selected_scan_data))
                st.info(f"Total signals: {signals_count}, including {included_count} in AI analysis (limited for performance)")
                
            with st.expander("Raw Spectrum Data Context"):
                st.text(context)
        else:
            st.info("No scans available for RAIDR analysis. Please upload a file (PNG, JSON, JSONL, CSV) or provide a valid Google Drive shareable link.")
    
    with tab3:
        st.header("Advanced SIGINT Analysis")
        if 'selected_scan_data' in locals() and selected_scan_data:
            context = prepare_llm_context(selected_scan_data, max_signals_per_scan=max_signals)
            
            if st.session_state.advanced_analysis_data['tactical_response']:
                tactical_resp = st.session_state.advanced_analysis_data['tactical_response']
                st.subheader("Tactical SIGINT Response:")
                st.markdown(f'<div class="tactical-text">{tactical_resp}</div>', unsafe_allow_html=True)
                
                st.subheader("Add a question about this response:")
                additional_question = st.text_area("Your question:", height=100)
                
                if st.button("Analyze This"):
                    if additional_question:
                        combined_query = f"RAIDR provided this tactical response:\n\n{tactical_resp}\n\nMy question about this: {additional_question}"
                        with st.spinner("Analyzing..."):
                            analysis_response = query_advanced_analysis(combined_query, context)
                            st.session_state.advanced_chat_history.append({"role": "user", "content": combined_query})
                            st.session_state.advanced_chat_history.append({"role": "assistant", "content": analysis_response})
                            st.session_state.advanced_analysis_data['tactical_response'] = ''
                        st.rerun()
                
                if st.button("Cancel"):
                    st.session_state.advanced_analysis_data['tactical_response'] = ''
                    st.rerun()
            
            else:
                chat_container = st.container()
                with chat_container:
                    for message in st.session_state.advanced_chat_history:
                        if message["role"] == "user":
                            st.markdown(f'<div class="message user-message">{message["content"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="message ai-message">{message["content"]}</div>', unsafe_allow_html=True)
                
                advanced_query = st.text_area("Ask your SIGINT analysis question:", height=100)
                if st.button("Send for Analysis"):
                    if advanced_query:
                        st.session_state.advanced_chat_history.append({"role": "user", "content": advanced_query})
                        with st.spinner("Analyzing..."):
                            advanced_response = query_advanced_analysis(advanced_query, context)
                            st.session_state.advanced_chat_history.append({"role": "assistant", "content": advanced_response})
                        st.rerun()
            
            if st.button("Clear Conversation History"):
                st.session_state.advanced_chat_history = []
                st.rerun()
                
            with st.expander("About Advanced Analysis"):
                st.markdown("""
                This tab provides access to a more conversational interface with GPT-4 Turbo, using a simple "SIGINT analyst" 
                prompt without the tactical radio format constraints. You can:
                
                1. Send tactical responses from the SIGINT tab for deeper analysis
                2. Ask detailed questions about signal patterns, threat assessments, or technical analysis
                3. Engage in a multi-turn conversation with the AI analyst
                """)
        else:
            st.info("No scans available for advanced analysis. Please upload a file (PNG, JSON, JSONL, CSV) or provide a valid Google Drive shareable link.")
    
    with tab4:
        st.header("TrellisWare Frequency Recommendation")
        if 'selected_scan_data' in locals() and selected_scan_data:
            scan = selected_scan_data[0]  # Use the first selected scan
            scan_data = scan["scan_data"]
            
            st.subheader("Analyzing Spectrum for TrellisWare Radio")
            with st.spinner("Analyzing interference with GPT-4 Turbo..."):
                recommended_freq, error_message = find_best_trellis_frequency(scan_data)
                if recommended_freq is not None:
                    st.markdown(f'<div class="lime-green">{recommended_freq:.3f}</div>', unsafe_allow_html=True)
                    st.success(f"Recommended Center Channel: {recommended_freq:.3f} MHz")
                else:
                    st.error(error_message)
            
            st.subheader("Scan Data Summary")
            signals = scan_data.get("detected_signals", [])
            if signals:
                signals_df = pd.DataFrame([
                    {
                        "Frequency (MHz)": signal.get("frequency", 0) / 1e6,
                        "Power (dBm)": signal.get("power_dbm", 0),
                        "Bandwidth (kHz)": signal.get("bandwidth", 0) / 1e3 if "bandwidth" in signal else "N/A",
                        "Type": signal.get("type", "unknown"),
                        "Threat Level": signal.get("threat_level", "unknown")
                    } for signal in signals
                ])
                st.dataframe(signals_df, use_container_width=True)
            else:
                st.info("No signals detected in this scan.")
        else:
            st.info("No scans available for Trellis analysis. Please upload a file (PNG, JSON, JSONL, CSV) or provide a valid Google Drive shareable link.")

if 'last_response' not in st.session_state:
    st.session_state.last_response = ""
if 'show_response' not in st.session_state:
    st.session_state.show_response = False

if __name__ == "__main__":
    main()