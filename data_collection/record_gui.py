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

if cv2 is not None:
    cv2_qt_plugin_dir = os.path.join(os.path.dirname(cv2.__file__), "qt", "plugins")
    configured_qt_plugin_dir = os.environ.get("QT_QPA_PLATFORM_PLUGIN_PATH")
    if configured_qt_plugin_dir and os.path.realpath(configured_qt_plugin_dir) == os.path.realpath(cv2_qt_plugin_dir):
        os.environ.pop("QT_QPA_PLATFORM_PLUGIN_PATH", None)

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

    class TrialProcessingThread(QtCore.QThread):
        completed = Signal(object)
        failed = Signal(str)

        def __init__(self, trial_buffers, bag_processes, sync_options, parent=None):
            super().__init__(parent)
            self.trial_buffers = trial_buffers
            self.bag_processes = bag_processes
            self.sync_options = sync_options

        def run(self):
            try:
                for process in self.bag_processes:
                    if process is not None:
                        stop_rosbag_recording(process)
                trial_data = synchronize_trial_data(
                    trial_buffers=self.trial_buffers,
                    **self.sync_options
                )
                self.completed.emit(trial_data)
            except Exception as exc:
                self.failed.emit(f"{type(exc).__name__}: {exc}")


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
            self.processing_thread = None

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
            QCheckBox {
                spacing: 9px;
                color: #f4f4f5;
                font-weight: 600;
            }
            QCheckBox:checked {
                color: #6ee7b7;
            }
            QCheckBox::indicator {
                width: 19px;
                height: 19px;
                border: 2px solid #d4d4d8;
                border-radius: 4px;
                background-color: #09090b;
            }
            QCheckBox::indicator:hover {
                border-color: #facc15;
            }
            QCheckBox::indicator:checked {
                border: 2px solid #d1fae5;
                background-color: #10b981;
            }
            QCheckBox::indicator:checked:hover {
                background-color: #059669;
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

            self.chk_camera_left = QCheckBox("Left")
            self.chk_camera_left.setChecked(True)
            self.edit_topic_left = QLineEdit(self.args.left_topic)
            layout_robot.addWidget(self.chk_camera_left, 1, 0)
            layout_robot.addWidget(self.edit_topic_left, 1, 1, 1, 2)

            self.chk_camera_right = QCheckBox("Right")
            self.chk_camera_right.setChecked(True)
            self.edit_topic_right = QLineEdit(self.args.right_topic)
            layout_robot.addWidget(self.chk_camera_right, 2, 0)
            layout_robot.addWidget(self.edit_topic_right, 2, 1, 1, 2)

            self.chk_wrist1 = QCheckBox("Wrist 1")
            self.chk_wrist1.setChecked("cam1" in self.args.wrist_cameras)
            self.edit_topic_wrist1 = QLineEdit(self.args.wrist1_topic)
            layout_robot.addWidget(self.chk_wrist1, 3, 0)
            layout_robot.addWidget(self.edit_topic_wrist1, 3, 1, 1, 2)

            self.chk_wrist2 = QCheckBox("Wrist 2")
            self.chk_wrist2.setChecked("cam2" in self.args.wrist_cameras)
            self.edit_topic_wrist2 = QLineEdit(self.args.wrist2_topic)
            layout_robot.addWidget(self.chk_wrist2, 4, 0)
            layout_robot.addWidget(self.edit_topic_wrist2, 4, 1, 1, 2)

            self.camera_controls = [
                ("left", self.chk_camera_left, self.edit_topic_left),
                ("right", self.chk_camera_right, self.edit_topic_right),
                ("wrist_1", self.chk_wrist1, self.edit_topic_wrist1),
                ("wrist_2", self.chk_wrist2, self.edit_topic_wrist2),
            ]
            for _, checkbox, topic_edit in self.camera_controls:
                checkbox.stateChanged.connect(self._on_camera_configuration_changed)
                topic_edit.editingFinished.connect(self._on_camera_configuration_changed)

            layout_robot.addWidget(QLabel("Image Resize:"), 5, 0)
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
            layout_robot.addLayout(layout_resize, 5, 1)

            left_layout.addWidget(grp_robot)

            # 4. Tolerances & Optional Rosbag Backups
            grp_opts = QGroupBox("Tolerances & Rosbag Backups")
            layout_opts = QGridLayout(grp_opts)
            layout_opts.setSpacing(8)

            layout_opts.addWidget(QLabel("State Tol (ms):"), 0, 0)
            self.spin_state_tol = QDoubleSpinBox()
            self.spin_state_tol.setRange(0.1, 1000.0)
            self.spin_state_tol.setDecimals(1)
            self.spin_state_tol.setSingleStep(1.0)
            self.spin_state_tol.setValue(float(self.args.state_tolerance) * 1000.0)
            layout_opts.addWidget(self.spin_state_tol, 0, 1)

            layout_opts.addWidget(QLabel("Image Tol (ms):"), 1, 0)
            self.spin_img_tol = QDoubleSpinBox()
            self.spin_img_tol.setRange(0.1, 1000.0)
            self.spin_img_tol.setDecimals(1)
            self.spin_img_tol.setSingleStep(1.0)
            self.spin_img_tol.setValue(float(self.args.image_tolerance) * 1000.0)
            layout_opts.addWidget(self.spin_img_tol, 1, 1)

            layout_opts.addWidget(QLabel("Max Interp Gap (ms):"), 2, 0)
            self.spin_max_interpolation_gap = QDoubleSpinBox()
            self.spin_max_interpolation_gap.setRange(2.0, 1000.0)
            self.spin_max_interpolation_gap.setDecimals(1)
            self.spin_max_interpolation_gap.setSingleStep(1.0)
            self.spin_max_interpolation_gap.setValue(float(self.args.max_interpolation_gap) * 1000.0)
            layout_opts.addWidget(self.spin_max_interpolation_gap, 2, 1)

            self.chk_record_bag = QCheckBox("Record Raw Rosbag")
            self.chk_record_bag.setChecked(bool(self.args.record_bag))
            layout_opts.addWidget(self.chk_record_bag, 3, 0, 1, 2)

            self.chk_force_bag = QCheckBox("Record Separate Force Bag")
            self.chk_force_bag.setChecked(bool(self.args.force_sensor))
            layout_opts.addWidget(self.chk_force_bag, 4, 0, 1, 2)

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

            # Side-by-side endoscope and wrist camera previews
            preview_panel = QWidget()
            preview_layout = QHBoxLayout(preview_panel)
            preview_layout.setContentsMargins(0, 0, 0, 0)
            preview_layout.setSpacing(8)

            self.grp_endoscope_preview = QGroupBox("Endoscope")
            endoscope_layout = QVBoxLayout(self.grp_endoscope_preview)
            self.lbl_preview = QLabel("Waiting for endoscope feed...")
            self.lbl_preview.setFixedSize(240, 180)
            self.lbl_preview.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
            self.lbl_preview.setStyleSheet("border: 1px dashed #52525b; border-radius: 6px; background: #121214;")
            endoscope_layout.addWidget(self.lbl_preview)
            preview_layout.addWidget(self.grp_endoscope_preview)

            self.grp_wrist_preview = QGroupBox("Wrist Camera")
            wrist_layout = QVBoxLayout(self.grp_wrist_preview)
            self.lbl_wrist_preview = QLabel("Waiting for wrist camera feed...")
            self.lbl_wrist_preview.setFixedSize(240, 180)
            self.lbl_wrist_preview.setAlignment(Qt.AlignmentFlag.AlignCenter if hasattr(Qt, 'AlignmentFlag') else Qt.AlignCenter)
            self.lbl_wrist_preview.setStyleSheet("border: 1px dashed #52525b; border-radius: 6px; background: #121214;")
            wrist_layout.addWidget(self.lbl_wrist_preview)
            preview_layout.addWidget(self.grp_wrist_preview)
            layout_ctrl_h.addWidget(preview_panel)

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
            self._apply_camera_configuration()
            self._log(f"Updated configuration to {self.combo_psm.currentText()}")

        def _selected_camera_names(self):
            return [name for name, checkbox, _ in self.camera_controls if checkbox.isChecked()]

        def _apply_camera_configuration(self):
            selected = set(self._selected_camera_names())
            wrist_cameras = []
            if "wrist_1" in selected: wrist_cameras.append("cam1")
            if "wrist_2" in selected: wrist_cameras.append("cam2")
            if self.node is not None:
                self.node.update_configuration(
                    psm=self.combo_psm.currentText(),
                    wrist_cameras=wrist_cameras,
                    left_topic=self.edit_topic_left.text().strip() if "left" in selected else "none",
                    right_topic=self.edit_topic_right.text().strip() if "right" in selected else "none",
                    wrist1_topic=self.edit_topic_wrist1.text().strip(),
                    wrist2_topic=self.edit_topic_wrist2.text().strip(),
                )
            return self._selected_camera_names()

        def _on_camera_configuration_changed(self):
            self._apply_camera_configuration()

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

            # 2. Update camera previews
            with self.node.preview_lock:
                preview_img = self.node.latest_preview_img
                wrist_preview_img = self.node.latest_wrist_preview_img
                wrist_preview_name = self.node.latest_wrist_preview_name
                wrist_preview_time = self.node.latest_wrist_preview_time

            self._set_preview_image(self.lbl_preview, preview_img, "Waiting for endoscope feed...")
            wrist_is_recent = wrist_preview_time and (time.monotonic() - wrist_preview_time) < 1.0
            if wrist_preview_img is not None and wrist_is_recent:
                camera_label = "Wrist 1" if wrist_preview_name == "wrist_1" else "Wrist 2"
                self.grp_wrist_preview.setTitle(camera_label)
                self._set_preview_image(self.lbl_wrist_preview, wrist_preview_img, "Waiting for wrist camera feed...")
            else:
                self.grp_wrist_preview.setTitle("Wrist Camera")
                self._set_preview_image(self.lbl_wrist_preview, None, "Waiting for wrist camera feed...")

        def _set_preview_image(self, label, image, waiting_text):
            if image is None or cv2 is None:
                label.setPixmap(QPixmap())
                label.setText(waiting_text)
                return

            height, width, channels = image.shape
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if self.args.image_encoding == 'bgr8' else image
            qimg = QImage(rgb.data, width, height, width * channels,
                          QImage.Format.Format_RGB888 if hasattr(QImage, 'Format') else QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg).scaled(
                label.size(),
                Qt.AspectRatioMode.KeepAspectRatio if hasattr(Qt, 'AspectRatioMode') else Qt.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation if hasattr(Qt, 'TransformationMode') else Qt.SmoothTransformation)
            label.setText("")
            label.setPixmap(pixmap)

        def _on_toggle_recording(self):
            if self.node is None:
                self._log("Cannot record: ROS 2 node is not initialized.")
                return

            if not self.node.is_recording:
                self._start_recording()
            else:
                self._stop_recording()

        def _start_recording(self):
            self.recording_camera_names = self._selected_camera_names()
            if not self.recording_camera_names:
                QMessageBox.warning(self, "Select a camera", "Select at least one camera topic to record.")
                return
            selected_controls = [entry for entry in self.camera_controls if entry[0] in self.recording_camera_names]
            if any(not topic_edit.text().strip() for _, _, topic_edit in selected_controls):
                QMessageBox.warning(self, "Camera topic required", "Enter a topic name for each selected camera.")
                return
            self._apply_camera_configuration()
            for _, checkbox, topic_edit in self.camera_controls:
                checkbox.setEnabled(False)
                topic_edit.setEnabled(False)

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

            topics = self.node.topic_names

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
            bag_processes = (self.bag_proc, self.force_bag_proc)
            self.bag_proc = None
            self.force_bag_proc = None

            self.btn_record.setText("PROCESSING TRIAL...")
            self.btn_record.setEnabled(False)
            self.lbl_record_status.setText("Processing & Synchronizing trial data...")
            self.lbl_review_stats.setText("Processing trial data. The recorder will remain available.")
            self.prog_bar.setRange(0, 0)
            self.btn_save.setEnabled(False)
            self.btn_discard.setEnabled(False)

            dur_mode = self.combo_duration_mode.currentText()
            req_seconds = self.spin_duration.value() if dur_mode == "Fixed Duration" else None
            resize = (self.spin_res_w.value(), self.spin_res_h.value()) if (self.spin_res_w.value() > 0 and self.spin_res_h.value() > 0) else None

            sync_options = dict(
                fps=self.spin_fps.value(),
                image_fps=self.spin_img_fps.value(),
                state_tolerance=self.spin_state_tol.value() / 1000.0,
                image_tolerance=self.spin_img_tol.value() / 1000.0,
                command_tolerance=float(self.args.command_tolerance),
                requested_seconds=req_seconds,
                resize=resize,
                command_mode=str(self.args.command_mode),
                wrist_cameras=self.args.wrist_cameras,
                wall_start_time=wall_start,
                camera_names=self.recording_camera_names,
                max_interpolation_gap=self.spin_max_interpolation_gap.value() / 1000.0
            )

            self.processing_thread = TrialProcessingThread(
                buffers, bag_processes, sync_options, self)
            self.processing_thread.completed.connect(self._on_trial_processed)
            self.processing_thread.failed.connect(self._on_trial_processing_failed)
            self.processing_thread.finished.connect(self.processing_thread.deleteLater)
            self.processing_thread.start()

        def _restore_after_processing(self):
            self.processing_thread = None
            self.prog_bar.setRange(0, 100)
            self.prog_bar.setValue(0)
            self.btn_record.setText("START RECORDING")
            self.btn_record.setObjectName("btnRecord")
            self.btn_record.setStyleSheet("background-color: #10b981; color: #fff;")
            self.btn_record.setEnabled(True)
            for _, checkbox, topic_edit in self.camera_controls:
                checkbox.setEnabled(True)
                topic_edit.setEnabled(True)

        def _on_trial_processing_failed(self, error_message):
            self._restore_after_processing()
            self.current_trial_data = None
            self.lbl_record_status.setText("Processing failed. Discard the trial or retry.")
            self.lbl_review_stats.setText(error_message)
            self.btn_save.setEnabled(False)
            self.btn_discard.setEnabled(True)
            self._log(f"Trial processing failed: {error_message}")

        def _on_trial_processed(self, trial_data):
            self._restore_after_processing()
            self.current_trial_data = trial_data
            camera_images = self.current_trial_data.get("images", {}) if self.current_trial_data else {}
            missing_cameras = [cam for cam, frames in camera_images.items() if not frames]
            has_camera_images = bool(camera_images) and not missing_cameras
            if (self.current_trial_data is None
                    or len(self.current_trial_data.get("qpos", [])) < 2
                    or not has_camera_images):
                self.lbl_record_status.setText("Trial failed: insufficient data or dropped frames.")
                if not has_camera_images:
                    camera_list = ", ".join(missing_cameras) if missing_cameras else "none configured"
                    self.lbl_record_status.setText("Trial failed: no camera frames were synchronized.")
                    self.lbl_review_stats.setText(f"No synchronized images for: {camera_list}. Check camera topics and retry.")
                else:
                    self.lbl_review_stats.setText("Synchronization returned too few samples. Discarding recommended.")
                self.btn_save.setEnabled(False)
                self.btn_discard.setEnabled(True)
                if not has_camera_images:
                    self._log(f"Synchronization failed: no images for {camera_list}.")
                else:
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
                image_encoding=self.args.image_encoding
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
            if self.processing_thread is not None and self.processing_thread.isRunning():
                QMessageBox.information(self, "Processing trial", "Please wait for trial processing to finish before closing.")
                event.ignore()
                return
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
