#!/usr/bin/env python3
# record_gui.py
#
# PyQt Graphical User Interface for dVRK teleoperation data collection.
# Separated from the backend recording code (record_dvrk_episodes.py) for modularity and readability.
#
# Features:
#   - Interactive parameter controls for all data collection arguments.
#   - Live topic monitoring table (reception status, active Hz rates, message counts).
#   - Live camera stream preview.
#   - Recording control with countdown timer and progress bar.
#   - Post-trial review with options to Save (HDF5 + optional bag) or Discard trial.

import os
import sys
import time
import shutil
import threading
from datetime import datetime

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

# Qt imports supporting PyQt5, PyQt6, PySide6, PySide2
QT_AVAILABLE = False
QT_LIB = None
try:
    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5.QtCore import Qt, QTimer, pyqtSignal as Signal
    from PyQt5.QtGui import QImage, QPixmap, QFont, QColor
    from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                 QHBoxLayout, QGridLayout, QLabel, QPushButton,
                                 QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
                                 QCheckBox, QFileDialog, QGroupBox, QProgressBar,
                                 QTextEdit, QTableWidget, QTableWidgetItem,
                                 QHeaderView, QSplitter, QFrame, QMessageBox)
    QT_AVAILABLE = True
    QT_LIB = "PyQt5"
except ImportError:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
        from PyQt6.QtCore import Qt, QTimer, pyqtSignal as Signal
        from PyQt6.QtGui import QImage, QPixmap, QFont, QColor
        from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                     QHBoxLayout, QGridLayout, QLabel, QPushButton,
                                     QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
                                     QCheckBox, QFileDialog, QGroupBox, QProgressBar,
                                     QTextEdit, QTableWidget, QTableWidgetItem,
                                     QHeaderView, QSplitter, QFrame, QMessageBox)
        QT_AVAILABLE = True
        QT_LIB = "PyQt6"
    except ImportError:
        try:
            from PySide6 import QtCore, QtGui, QtWidgets
            from PySide6.QtCore import Qt, QTimer, Signal
            from PySide6.QtGui import QImage, QPixmap, QFont, QColor
            from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                                         QHBoxLayout, QGridLayout, QLabel, QPushButton,
                                         QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox,
                                         QCheckBox, QFileDialog, QGroupBox, QProgressBar,
                                         QTextEdit, QTableWidget, QTableWidgetItem,
                                         QHeaderView, QSplitter, QFrame, QMessageBox)
            QT_AVAILABLE = True
            QT_LIB = "PySide6"
        except ImportError:
            QT_AVAILABLE = False
            QT_LIB = None

# Backend recorder functions
from record_dvrk_episodes import (
    DVRKRecorder,
    synchronize_trial_data,
    save_episode,
    derive_topic_names,
    get_next_episode_idx,
    start_rosbag_recording,
    stop_rosbag_recording,
    parse_args,
    ROS_AVAILABLE,
)

if ROS_AVAILABLE:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor


if QT_AVAILABLE:

    class RecordMainWindow(QMainWindow):
        """
        PyQt Graphical User Interface for dVRK teleoperation data collection.
        Allows controlling all recording arguments as inputs, monitoring real-time topic rates,
        and choosing to Save or Discard each trial.
        """

        def __init__(self, node, default_args):
            super().__init__()
            self.node = node
            self.args = default_args

            self.current_trial_data = None
            self.current_bag_path = None
            self.current_force_bag_path = None
            self.bag_proc = None
            self.force_bag_proc = None

            self.setWindowTitle("dVRK Episode Recorder & Data Collection")
            self.resize(1200, 800)
            self._apply_dark_theme()

            self._build_ui()

            # Timer for live UI monitoring (topic rates & camera preview)
            self.monitor_timer = QTimer(self)
            self.monitor_timer.timeout.connect(self._update_monitoring)
            self.monitor_timer.start(50)  # 20 Hz UI refresh

            # Timer for fixed-duration trial countdown
            self.record_timer = QTimer(self)
            self.record_timer.timeout.connect(self._on_record_timer_tick)
            self.record_elapsed = 0.0

        def _apply_dark_theme(self):
            qss = """
            QMainWindow, QWidget {
                background-color: #18181b;
                color: #f4f4f5;
                font-family: 'Segoe UI', -apple-system, Roboto, sans-serif;
                font-size: 13px;
            }
            QGroupBox {
                border: 1px solid #3f3f46;
                border-radius: 8px;
                margin-top: 12px;
                padding-top: 14px;
                font-weight: 600;
                color: #60a5fa;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 5px;
                padding: 5px 8px;
                color: #fafafa;
                selection-background-color: #3b82f6;
            }
            QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
                border: 1px solid #3b82f6;
            }
            QPushButton {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 7px 14px;
                font-weight: 600;
                color: #f4f4f5;
            }
            QPushButton:hover {
                background-color: #3f3f46;
            }
            QPushButton#btnRecord {
                background-color: #10b981;
                border: none;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                color: #ffffff;
            }
            QPushButton#btnRecord:hover {
                background-color: #059669;
            }
            QPushButton#btnStop {
                background-color: #ef4444;
                border: none;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 20px;
                color: #ffffff;
            }
            QPushButton#btnStop:hover {
                background-color: #dc2626;
            }
            QPushButton#btnSave {
                background-color: #10b981;
                border: none;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 18px;
                color: #ffffff;
            }
            QPushButton#btnSave:hover {
                background-color: #059669;
            }
            QPushButton#btnDiscard {
                background-color: #ef4444;
                border: none;
                font-size: 14px;
                font-weight: bold;
                padding: 8px 18px;
                color: #ffffff;
            }
            QPushButton#btnDiscard:hover {
                background-color: #dc2626;
            }
            QProgressBar {
                border: 1px solid #3f3f46;
                border-radius: 5px;
                text-align: center;
                background-color: #27272a;
                color: #f4f4f5;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 4px;
            }
            QTableWidget {
                background-color: #27272a;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                gridline-color: #3f3f46;
            }
            QHeaderView::section {
                background-color: #1f1f23;
                padding: 4px;
                border: 1px solid #3f3f46;
                font-weight: bold;
                color: #a1a1aa;
            }
            QTextEdit {
                background-color: #1f1f23;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                font-family: Consolas, monospace;
                font-size: 12px;
                color: #d4d4d8;
            }
            """
            self.setStyleSheet(qss)

        def _build_ui(self):
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QHBoxLayout(central)
            main_layout.setContentsMargins(12, 12, 12, 12)
            main_layout.setSpacing(12)

            splitter = QSplitter(Qt.Orientation.Horizontal if hasattr(Qt, 'Orientation') else Qt.Horizontal)
            main_layout.addWidget(splitter)

            # ── Left Column: Parameters & Settings (Inputs / Arguments) ──
            left_pane = QWidget()
            left_layout = QVBoxLayout(left_pane)
            left_layout.setContentsMargins(0, 0, 0, 0)
            left_layout.setSpacing(10)

            # 1. Output & Episode
            grp_storage = QGroupBox("Dataset & Output")
            layout_storage = QGridLayout(grp_storage)
            layout_storage.setSpacing(8)

            layout_storage.addWidget(QLabel("Output Dir:"), 0, 0)
            self.edit_output_dir = QLineEdit(self.args.output_dir)
            layout_storage.addWidget(self.edit_output_dir, 0, 1)
            btn_browse = QPushButton("Browse...")
            btn_browse.clicked.connect(self._on_browse_output)
            layout_storage.addWidget(btn_browse, 0, 2)

            layout_storage.addWidget(QLabel("Episode Index:"), 1, 0)
            self.spin_episode_idx = QSpinBox()
            self.spin_episode_idx.setRange(0, 99999)
            self.spin_episode_idx.setValue(get_next_episode_idx(self.args.output_dir))
            layout_storage.addWidget(self.spin_episode_idx, 1, 1)

            left_layout.addWidget(grp_storage)

            # 2. Timing & Frequency Parameters
            grp_timing = QGroupBox("Timing & Sampling Rates")
            layout_timing = QGridLayout(grp_timing)
            layout_timing.setSpacing(8)

            layout_timing.addWidget(QLabel("Duration Mode:"), 0, 0)
            self.combo_duration_mode = QComboBox()
            self.combo_duration_mode.addItems(["Fixed Duration", "Manual Start/Stop"])
            layout_timing.addWidget(self.combo_duration_mode, 0, 1)

            layout_timing.addWidget(QLabel("Duration (s):"), 1, 0)
            self.spin_duration = QDoubleSpinBox()
            self.spin_duration.setRange(1.0, 300.0)
            self.spin_duration.setValue(float(self.args.seconds_per_episode))
            self.spin_duration.setSingleStep(1.0)
            layout_timing.addWidget(self.spin_duration, 1, 1)

            layout_timing.addWidget(QLabel("State FPS (Hz):"), 2, 0)
            self.spin_fps = QDoubleSpinBox()
            self.spin_fps.setRange(1.0, 1000.0)
            self.spin_fps.setValue(float(self.args.fps))
            layout_timing.addWidget(self.spin_fps, 2, 1)

            layout_timing.addWidget(QLabel("Camera FPS (Hz):"), 3, 0)
            self.spin_img_fps = QDoubleSpinBox()
            self.spin_img_fps.setRange(1.0, 60.0)
            self.spin_img_fps.setValue(float(self.args.image_fps))
            layout_timing.addWidget(self.spin_img_fps, 3, 1)

            layout_timing.addWidget(QLabel("Warmup (s):"), 4, 0)
            self.spin_warmup = QDoubleSpinBox()
            self.spin_warmup.setRange(0.0, 10.0)
            self.spin_warmup.setValue(float(self.args.warmup_sec))
            layout_timing.addWidget(self.spin_warmup, 4, 1)

            left_layout.addWidget(grp_timing)

            # 3. Robot & Camera Configuration
            grp_robot = QGroupBox("Robot & Camera Configuration")
            layout_robot = QGridLayout(grp_robot)
            layout_robot.setSpacing(8)

            layout_robot.addWidget(QLabel("PSM Arm:"), 0, 0)
            self.combo_psm = QComboBox()
            self.combo_psm.addItems(["PSM2", "PSM1"])
            self.combo_psm.setCurrentText(self.args.psm)
            self.combo_psm.currentIndexChanged.connect(self._on_psm_changed)
            layout_robot.addWidget(self.combo_psm, 0, 1)

            layout_robot.addWidget(QLabel("Arm Joints:"), 1, 0)
            self.spin_arm_joints = QSpinBox()
            self.spin_arm_joints.setRange(1, 10)
            self.spin_arm_joints.setValue(int(self.args.arm_joints))
            layout_robot.addWidget(self.spin_arm_joints, 1, 1)

            layout_robot.addWidget(QLabel("Wrist Cams:"), 2, 0)
            layout_wrist = QHBoxLayout()
            self.chk_wrist1 = QCheckBox("Cam 1")
            self.chk_wrist1.setChecked("cam1" in self.args.wrist_cameras)
            self.chk_wrist2 = QCheckBox("Cam 2")
            self.chk_wrist2.setChecked("cam2" in self.args.wrist_cameras)
            self.chk_wrist1.stateChanged.connect(self._on_wrist_changed)
            self.chk_wrist2.stateChanged.connect(self._on_wrist_changed)
            layout_wrist.addWidget(self.chk_wrist1)
            layout_wrist.addWidget(self.chk_wrist2)
            layout_robot.addLayout(layout_wrist, 2, 1)

            layout_robot.addWidget(QLabel("Image Resize:"), 3, 0)
            layout_resize = QHBoxLayout()
            self.spin_res_w = QSpinBox()
            self.spin_res_w.setRange(0, 1920)
            self.spin_res_w.setValue(int(self.args.resize_w))
            self.spin_res_h = QSpinBox()
            self.spin_res_h.setRange(0, 1080)
            self.spin_res_h.setValue(int(self.args.resize_h))
            layout_resize.addWidget(self.spin_res_w)
            layout_resize.addWidget(QLabel("x"))
            layout_resize.addWidget(self.spin_res_h)
            layout_robot.addLayout(layout_resize, 3, 1)

            layout_robot.addWidget(QLabel("Encoding:"), 4, 0)
            self.combo_encoding = QComboBox()
            self.combo_encoding.addItems(["bgr8", "rgb8"])
            self.combo_encoding.setCurrentText(self.args.image_encoding)
            layout_robot.addWidget(self.combo_encoding, 4, 1)

            left_layout.addWidget(grp_robot)

            # 4. Tolerances & Optional Rosbag Backups
            grp_opts = QGroupBox("Tolerances & Rosbag Backups")
            layout_opts = QGridLayout(grp_opts)
            layout_opts.setSpacing(8)

            layout_opts.addWidget(QLabel("State Tol (s):"), 0, 0)
            self.spin_state_tol = QDoubleSpinBox()
            self.spin_state_tol.setRange(0.001, 1.0)
            self.spin_state_tol.setValue(float(self.args.state_tolerance))
            layout_opts.addWidget(self.spin_state_tol, 0, 1)

            layout_opts.addWidget(QLabel("Image Tol (s):"), 1, 0)
            self.spin_img_tol = QDoubleSpinBox()
            self.spin_img_tol.setRange(0.001, 1.0)
            self.spin_img_tol.setValue(float(self.args.image_tolerance))
            layout_opts.addWidget(self.spin_img_tol, 1, 1)

            self.chk_record_bag = QCheckBox("Record Raw Rosbag")
            self.chk_record_bag.setChecked(bool(self.args.record_bag))
            layout_opts.addWidget(self.chk_record_bag, 2, 0, 1, 2)

            self.chk_force_bag = QCheckBox("Record Separate Force Bag")
            self.chk_force_bag.setChecked(bool(self.args.force_sensor))
            layout_opts.addWidget(self.chk_force_bag, 3, 0, 1, 2)

            left_layout.addWidget(grp_opts)
            left_layout.addStretch()

            splitter.addWidget(left_pane)

            # ── Right Column: Monitoring, Controls & Review ──
            right_pane = QWidget()
            right_layout = QVBoxLayout(right_pane)
            right_layout.setContentsMargins(0, 0, 0, 0)
            right_layout.setSpacing(10)

            # Live Topic Health Table
            grp_topics = QGroupBox("Live Topic Status & Reception Frequency")
            layout_topics = QVBoxLayout(grp_topics)
            self.table_topics = QTableWidget()
            self.table_topics.setColumnCount(4)
            self.table_topics.setHorizontalHeaderLabels(["Status", "Topic Name", "Live Rate (Hz)", "Total Messages"])
            header = self.table_topics.horizontalHeader()
            if hasattr(header, 'setSectionResizeMode'):
                header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch if hasattr(QHeaderView, 'ResizeMode') else QHeaderView.Stretch)
            layout_topics.addWidget(self.table_topics)
            right_layout.addWidget(grp_topics, stretch=2)

            # Camera Preview & Recording Control
            panel_control = QWidget()
            layout_ctrl_h = QHBoxLayout(panel_control)
            layout_ctrl_h.setContentsMargins(0, 0, 0, 0)
            layout_ctrl_h.setSpacing(12)

            # Camera feed preview box
            self.lbl_preview = QLabel("Waiting for camera feed...")
            self.lbl_preview.setFixedSize(320, 240)
            self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
            self.lbl_preview.setStyleSheet("border: 1px dashed #3f3f46; border-radius: 6px; background: #121214;")
            layout_ctrl_h.addWidget(self.lbl_preview)

            # Recording action bar
            ctrl_box = QWidget()
            layout_actions = QVBoxLayout(ctrl_box)
            layout_actions.setContentsMargins(0, 0, 0, 0)

            self.btn_record = QPushButton("START RECORDING")
            self.btn_record.setObjectName("btnRecord")
            self.btn_record.setFixedHeight(48)
            self.btn_record.clicked.connect(self._on_toggle_recording)
            layout_actions.addWidget(self.btn_record)

            self.lbl_record_status = QLabel("Ready to record episode.")
            self.lbl_record_status.setStyleSheet("font-size: 14px; font-weight: bold; color: #a1a1aa;")
            layout_actions.addWidget(self.lbl_record_status)

            self.prog_bar = QProgressBar()
            self.prog_bar.setRange(0, 100)
            self.prog_bar.setValue(0)
            self.prog_bar.setFixedHeight(24)
            layout_actions.addWidget(self.prog_bar)

            layout_actions.addStretch()
            layout_ctrl_h.addWidget(ctrl_box, stretch=1)
            right_layout.addWidget(panel_control)

            # Trial Review & Decision Card (Save or Discard)
            self.grp_review = QGroupBox("Trial Review (Save / Discard Decision)")
            layout_review = QVBoxLayout(self.grp_review)
            self.lbl_review_stats = QLabel("No trial completed yet.")
            self.lbl_review_stats.setStyleSheet("font-size: 13px; color: #e4e4e7;")
            layout_review.addWidget(self.lbl_review_stats)

            layout_review_btns = QHBoxLayout()
            self.btn_save = QPushButton("SAVE TRIAL (HDF5)")
            self.btn_save.setObjectName("btnSave")
            self.btn_save.setEnabled(False)
            self.btn_save.clicked.connect(self._on_save_trial)

            self.btn_discard = QPushButton("DISCARD TRIAL")
            self.btn_discard.setObjectName("btnDiscard")
            self.btn_discard.setEnabled(False)
            self.btn_discard.clicked.connect(self._on_discard_trial)

            layout_review_btns.addWidget(self.btn_save)
            layout_review_btns.addWidget(self.btn_discard)
            layout_review.addLayout(layout_review_btns)
            right_layout.addWidget(self.grp_review)

            # Activity Log
            self.txt_log = QTextEdit()
            self.txt_log.setReadOnly(True)
            self.txt_log.setFixedHeight(120)
            right_layout.addWidget(self.txt_log)

            splitter.addWidget(right_pane)
            splitter.setSizes([380, 820])

            self._log("dVRK Data Collection GUI initialized.")

        def _log(self, msg):
            ts = datetime.now().strftime("%H:%M:%S")
            self.txt_log.append(f"[{ts}] {msg}")

        def _on_browse_output(self):
            folder = QFileDialog.getExistingDirectory(self, "Select Output Directory", self.edit_output_dir.text())
            if folder:
                self.edit_output_dir.setText(folder)
                self.spin_episode_idx.setValue(get_next_episode_idx(folder))

        def _on_psm_changed(self):
            new_psm = self.combo_psm.currentText()
            wrist_cams = []
            if self.chk_wrist1.isChecked(): wrist_cams.append('cam1')
            if self.chk_wrist2.isChecked(): wrist_cams.append('cam2')
            if self.node is not None:
                self.node.update_configuration(psm=new_psm, wrist_cameras=wrist_cams)
            self._log(f"Updated configuration to {new_psm}")

        def _on_wrist_changed(self):
            self._on_psm_changed()

        def _update_monitoring(self):
            if self.node is None:
                return

            # 1. Update topic table
            stats = self.node.get_live_stats()
            self.table_topics.setRowCount(len(stats))
            for row, (topic, data) in enumerate(sorted(stats.items())):
                status_text = "🟢 Active" if data['active'] else "⚪ Waiting"
                item_status = QTableWidgetItem(status_text)
                item_status.setTextAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
                item_topic = QTableWidgetItem(topic)
                item_rate = QTableWidgetItem(f"{data['rate_hz']:.1f} Hz" if data['active'] else "--")
                item_rate.setTextAlignment(Qt.AlignmentFlag.AlignRight if hasattr(Qt, 'AlignmentFlag') else Qt.AlignRight)
                item_count = QTableWidgetItem(str(data['count']))
                item_count.setTextAlignment(Qt.AlignmentFlag.AlignRight if hasattr(Qt, 'AlignmentFlag') else Qt.AlignRight)

                self.table_topics.setItem(row, 0, item_status)
                self.table_topics.setItem(row, 1, item_topic)
                self.table_topics.setItem(row, 2, item_rate)
                self.table_topics.setItem(row, 3, item_count)

            # 2. Update camera preview
            with self.node.preview_lock:
                preview_img = self.node.latest_preview_img

            if preview_img is not None and cv2 is not None:
                h, w, c = preview_img.shape
                rgb = cv2.cvtColor(preview_img, cv2.COLOR_BGR2RGB) if self.combo_encoding.currentText() == 'bgr8' else preview_img
                qimg = QImage(rgb.data, w, h, w * c, QImage.Format.Format_RGB888 if hasattr(QImage, 'Format') else QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qimg).scaled(320, 240,
                    Qt.AspectRatioMode.KeepAspectRatio if hasattr(Qt, 'AspectRatioMode') else Qt.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation if hasattr(Qt, 'TransformationMode') else Qt.SmoothTransformation)
                self.lbl_preview.setPixmap(pixmap)

        def _on_toggle_recording(self):
            if self.node is None:
                self._log("Cannot record: ROS 2 node is not initialized.")
                return

            if not self.node.is_recording:
                self._start_recording()
            else:
                self._stop_recording()

        def _start_recording(self):
            idx = self.spin_episode_idx.value()
            self._log(f"Starting recording trial episode_{idx:03d} ...")

            # Reset review card
            self.btn_save.setEnabled(False)
            self.btn_discard.setEnabled(False)
            self.lbl_review_stats.setText("Recording in progress...")

            # Rosbag backups if selected
            self.current_bag_path = None
            self.current_force_bag_path = None
            self.bag_proc = None
            self.force_bag_proc = None

            psm = self.combo_psm.currentText()
            wrist_cams = []
            if self.chk_wrist1.isChecked(): wrist_cams.append('cam1')
            if self.chk_wrist2.isChecked(): wrist_cams.append('cam2')
            topics = derive_topic_names(psm=psm, wrist_cameras=wrist_cams)

            if self.chk_record_bag.isChecked():
                bag_dir = self.args.bag_dir
                os.makedirs(bag_dir, exist_ok=True)
                self.current_bag_path = os.path.join(bag_dir, f'episode_{idx:03d}_bag')
                bag_topics = list(topics.values())
                self.bag_proc = start_rosbag_recording(self.current_bag_path, bag_topics)

            if self.chk_force_bag.isChecked():
                force_dir = self.args.force_bag_dir
                os.makedirs(force_dir, exist_ok=True)
                self.current_force_bag_path = os.path.join(force_dir, f'episode_{idx:03d}_force')
                force_topics = [
                    topics["measured_cf_topic"],
                    topics["gravity_topic"],
                    topics["psm_arm_topic"],
                    topics["spatial_cf_topic"],
                    topics["body_cf_topic"],
                ]
                self.force_bag_proc = start_rosbag_recording(self.current_force_bag_path, force_topics)

            # Start node intake
            self.node.start_recording()
            self.btn_record.setText("STOP RECORDING")
            self.btn_record.setObjectName("btnStop")
            self.btn_record.setStyleSheet("background-color: #ef4444; color: #fff;")

            self.record_elapsed = 0.0
            self.record_timer.start(100)  # 10 Hz progress tick

        def _on_record_timer_tick(self):
            self.record_elapsed += 0.1
            mode = self.combo_duration_mode.currentText()
            total_dur = self.spin_duration.value()

            if mode == "Fixed Duration":
                pct = min(100, int((self.record_elapsed / total_dur) * 100))
                self.prog_bar.setValue(pct)
                self.lbl_record_status.setText(f"Recording: {self.record_elapsed:.1f}s / {total_dur:.1f}s")
                if self.record_elapsed >= total_dur:
                    self._stop_recording()
            else:
                self.prog_bar.setValue((int(self.record_elapsed * 20) % 100))
                self.lbl_record_status.setText(f"Recording (Manual): {self.record_elapsed:.1f}s elapsed")

        def _stop_recording(self):
            self.record_timer.stop()
            buffers, wall_start = self.node.stop_recording()

            # Stop rosbag processes
            if self.bag_proc:
                stop_rosbag_recording(self.bag_proc)
                self.bag_proc = None
            if self.force_bag_proc:
                stop_rosbag_recording(self.force_bag_proc)
                self.force_bag_proc = None

            self.btn_record.setText("START RECORDING")
            self.btn_record.setObjectName("btnRecord")
            self.btn_record.setStyleSheet("background-color: #10b981; color: #fff;")
            self.lbl_record_status.setText("Processing & Synchronizing trial data...")

            # Run nearest-timestep synchronizer at the specified frequency
            dur_mode = self.combo_duration_mode.currentText()
            req_seconds = self.spin_duration.value() if dur_mode == "Fixed Duration" else None
            resize = (self.spin_res_w.value(), self.spin_res_h.value()) if (self.spin_res_w.value() > 0 and self.spin_res_h.value() > 0) else None

            wrist_cams = []
            if self.chk_wrist1.isChecked(): wrist_cams.append('cam1')
            if self.chk_wrist2.isChecked(): wrist_cams.append('cam2')

            self.current_trial_data = synchronize_trial_data(
                trial_buffers=buffers,
                fps=self.spin_fps.value(),
                image_fps=self.spin_img_fps.value(),
                state_tolerance=self.spin_state_tol.value(),
                image_tolerance=self.spin_img_tol.value(),
                command_tolerance=float(self.args.command_tolerance),
                requested_seconds=req_seconds,
                resize=resize,
                command_mode=str(self.args.command_mode),
                wrist_cameras=wrist_cams,
                wall_start_time=wall_start
            )

            if self.current_trial_data is None or len(self.current_trial_data.get("qpos", [])) < 2:
                self.lbl_record_status.setText("Trial failed: insufficient data or dropped frames.")
                self.lbl_review_stats.setText("⚠️ Synchronization returned too few samples. Discarding recommended.")
                self.btn_save.setEnabled(False)
                self.btn_discard.setEnabled(True)
                self._log("Synchronization failed: insufficient samples.")
                return

            meta = self.current_trial_data["meta"]
            captured = meta["captured_steps"]
            requested = meta["requested_steps"]
            dropped = meta["dropped_steps"]
            ratio = meta["capture_ratio"] * 100.0

            stats_str = (f"<b>Trial #{self.spin_episode_idx.value()} Review:</b> "
                         f"Captured <b>{captured}</b> / {requested} steps "
                         f"({ratio:.1f}% ratio), <b>{dropped}</b> dropped.")
            self.lbl_review_stats.setText(stats_str)
            self.lbl_record_status.setText("Trial Completed! Choose to Save or Discard below.")
            self.btn_save.setEnabled(True)
            self.btn_discard.setEnabled(True)
            self._log(f"Trial #{self.spin_episode_idx.value()} ready: {captured}/{requested} steps ({ratio:.1f}%).")

        def _on_save_trial(self):
            if self.current_trial_data is None:
                return

            idx = self.spin_episode_idx.value()
            output_dir = self.edit_output_dir.text()
            save_path = os.path.join(output_dir, f'episode_{idx:03d}.hdf5')

            resize = (self.spin_res_w.value(), self.spin_res_h.value()) if (self.spin_res_w.value() > 0 and self.spin_res_h.value() > 0) else None

            save_episode(
                save_path=save_path,
                data=self.current_trial_data,
                episode_idx=idx,
                topic_names=self.node.topic_names,
                resize=resize,
                image_encoding=self.combo_encoding.currentText()
            )

            self._log(f"✅ Saved Episode #{idx} -> {save_path}")
            self.spin_episode_idx.setValue(idx + 1)

            # Reset review panel
            self.current_trial_data = None
            self.btn_save.setEnabled(False)
            self.btn_discard.setEnabled(False)
            self.lbl_record_status.setText(f"Episode #{idx} saved. Ready for next trial.")
            self.prog_bar.setValue(0)

        def _on_discard_trial(self):
            idx = self.spin_episode_idx.value()
            self._log(f"❌ Discarded Trial #{idx}.")

            # Clean up rosbag folders if created
            if self.current_bag_path and os.path.exists(self.current_bag_path):
                shutil.rmtree(self.current_bag_path, ignore_errors=True)
            if self.current_force_bag_path and os.path.exists(self.current_force_bag_path):
                shutil.rmtree(self.current_force_bag_path, ignore_errors=True)

            self.current_trial_data = None
            self.btn_save.setEnabled(False)
            self.btn_discard.setEnabled(False)
            self.lbl_record_status.setText(f"Trial #{idx} discarded. Ready to retry.")
            self.lbl_review_stats.setText("Trial discarded. Index retained for re-attempt.")
            self.prog_bar.setValue(0)

        def closeEvent(self, event):
            self.monitor_timer.stop()
            self.record_timer.stop()
            if self.node is not None and self.node.is_recording:
                self.node.stop_recording()
            event.accept()


def launch_gui(args, node=None):
    """Launch the PyQt application with the recorder node."""
    if not QT_AVAILABLE:
        print(f"[ERROR] Cannot launch GUI: No PyQt5, PyQt6, or PySide installed.")
        sys.exit(1)

    print(f"[INFO] Launching PyQt GUI ({QT_LIB})...")
    app = QApplication.instance() or QApplication(sys.argv)

    # If node not passed, create and start background ROS executor
    executor = None
    if node is None and ROS_AVAILABLE:
        rclpy.init()
        node = DVRKRecorder(
            psm=args.psm,
            wrist_cameras=args.wrist_cameras,
            left_topic=args.left_topic,
            right_topic=args.right_topic,
            wrist1_topic=args.wrist1_topic,
            wrist2_topic=args.wrist2_topic,
            arm_joints=args.arm_joints,
            jaw_joints=args.jaw_joints,
            image_encoding=args.image_encoding,
        )
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        ros_thread = threading.Thread(target=executor.spin, daemon=True)
        ros_thread.start()

    window = RecordMainWindow(node, args)
    window.show()
    ret = app.exec_() if hasattr(app, 'exec_') else app.exec()

    if ROS_AVAILABLE:
        try:
            rclpy.shutdown()
        except Exception:
            pass
    sys.exit(ret)


if __name__ == "__main__":
    args = parse_args()
    launch_gui(args)
