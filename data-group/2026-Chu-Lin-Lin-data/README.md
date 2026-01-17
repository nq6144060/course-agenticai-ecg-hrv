# Data: Multi-Modal Physiological Dataset for Personalized HRV Analysis

**Group:** 2026-Chu-Lin-Lin  
**Authors:** LIN, CHIH-YI; CHU, YEN-CHIEH; LIN, WEN-HSIN  
**License:** CC-BY-4.0 (this documentation); Original dataset CC-BY-4.0

## Dataset Overview

### Source

This dataset contains synchronized multi-modal physiological signals (PPG and ECG) and camera frame references, designed to bridge the personalization gap in longitudinal monitoring.

**Original Source:** Internally collected by the 2026-Chu-Lin-Lin research team for the Agentic AI course project.  
**Download at:** `data-group/2026-Chu-Lin-Lin-data/processed/`

### License

**Original License:** CC-BY-4.0

Citations required for using this data:

1. LIN, C. Y., CHU, Y. C., & LIN, W. H. (2026). *Multi-Modal Physiological Dataset for Personalized HRV Analysis*. Course Project for Agentic AI.
2. 2026-Chu-Lin-Lin Group. "Bridging the Personalization Gap in Longitudinal Physiological Monitoring via Agentic AI," Case Brief, 2026.

### Statistics

| Property | Value |
|----------|-------|
| Subjects | 2 (ID: 595, 809) |
| Demographics | Anonymized; Healthy adults |
| Data Length | ~5-10 minutes per session |
| Sampling Rate | ~250 Hz (Extended Polling) |
| Total Files | 16 |
| Modalities | 2 (PPG, ECG) |
| Conditions | 4 (Static, Bike, Speak, Rotate) |



### Format

| File | Description | Format |
|------|-------------|--------|
| `processed/*.csv` | Time-aligned PPG/ECG/Camera Frame data | CSV, UTF-8 |

#### processed/*.csv Columns

The processed data files are headerless CSVs. The column mapping is as follows:

| Column | Type | Description |
|--------|------|-------------|
| **1** | int | **Camera Frame Count** (Video synchronization reference) |
| **2** | float | System Timestamp (ms) |
| **3** | float | **PPG Data** (Photoplethysmogram / Pulse signal) |
| **4** | float | **ECG Data** (Electrocardiogram / Electrical heart signal) |
| **5** | float | Auxiliary Sensor Data 1 (Accelerometer Axis) |
| **6** | float | Auxiliary Sensor Data 2 (Accelerometer Axis) |

#### Labels

| Category | Mapping | Description |
|----------|---------|-------------|
| **Subject ID** | `595`, `809` | Anonymized identifier for the individual |
| **Activity** | `static`, `bike`, `speak`, `rotate` | Physiological state / context |
| **Intensity** | `level1`, `level3`, `level5` | Level of physical or cognitive exertion |

> **Note:** Labels are encoded within the filename of each CSV to enable Agentic AI context recognition (e.g., `595_bike_level5_...csv`).

### Preprocessing

1. **Hardware Synchronization:** Signals are aligned using the Camera Frame Count (Column 1) to ensure physiological data maps to the visual ground truth.
2. **Binary-to-CSV Conversion:** Raw binary polling data was converted to decimal CSV format with high-precision timestamp preservation.
3. **State Segmentation:** Data is organized by activity intensity to facilitate context-aware AI training for personalized baselines.
4. **Noise Preservation:** Motion artifacts in 'Bike' and 'Rotate' files are intentionally preserved to test adaptive R-peak detection algorithms.

### Privacy

This dataset contains no personally identifiable information (PII). Subject IDs (595, 809) are randomized identifiers. All data was collected with informed consent and follows the privacy guidelines for academic research.

### Usage

This dataset is designed for use in an **Agentic AI Adaptive Analysis Loop**. Below is the code example for loading and processing your specific multi-modal data format:

```python
import pandas as pd

# Define headers matching your data structure
headers = ['frame', 'timestamp', 'ppg', 'ecg', 'aux1', 'aux2']

# Load processed data (Example: Subject 809 Static Level 1)
file_path = 'processed/809_static_level1_20260113_232205_extended_poll.bin_align.csv'
data = pd.read_csv(file_path, names=headers)

# 1. Access PPG and ECG for personalized HRV analysis
ppg_signal = data['ppg']
ecg_signal = data['ecg']

# 2. Use 'frame' to synchronize with external video data
sync_frame = data['frame']

# Agent Logic Tip: 
# The system can detect activity state from the filename and 
# select the appropriate filter for the 'ecg' column.