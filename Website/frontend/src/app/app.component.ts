import { Component, OnInit, OnDestroy, HostListener } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { Subscription, interval } from 'rxjs';
import { RoverApiService, SystemStatus, DetectionRecord } from './services/rover-api.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.css']
})
export class AppComponent implements OnInit, OnDestroy {
  title = 'AgriRover AI — Precision Agriculture Smart Rover';
  // EXACTLY FIVE PAGES
  activeTab: 'dashboard' | 'rover' | 'data' | 'analytics' | 'settings' = 'dashboard';
  mobileMenuOpen: boolean = false;

  // Live Subsystem Status
  status: SystemStatus | null = null;
  latestDetection: DetectionRecord | null = null;
  historyList: DetectionRecord[] = [];
  filteredHistoryList: DetectionRecord[] = [];
  statistics: any = null;
  diseaseInfoList: any[] = [];

  // Rover Movement
  roverSpeed: number = 75;
  activeMovement: string = 'STOPPED';
  isMoving: boolean = false;

  // Camera Live Stream & Video Controls
  liveStreamActive: boolean = true;
  streamError: boolean = false;
  streamLoaded: boolean = false;
  autoDetectEnabled: boolean = true;
  autoInterval: number = 3.0;
  streamTimestamp: number = Date.now();
  useSingleFrameMode: boolean = false;
  frameTimer: any = null;
  liveFpsDisplay: number = 0;
  frameCount: number = 0;
  lastFpsCalcTime: number = Date.now();

  // Sprayer State
  sprayDuration: number = 3.0;
  sprayerActive: boolean = false;
  showSprayConfirmModal: boolean = false;
  sprayTargetDisease: string = '';

  // Daily Data Search & Filtering
  dataSearchQuery: string = '';
  dataFilterDisease: string = 'ALL';
  dataMinConfidence: number = 0;

  // Settings
  apiEndpoint: string = '';
  confidenceThreshold: number = 70;
  selectedCloudPreset: string = 'render';

  // UI Alerts
  alertMessage: string | null = null;
  alertType: 'success' | 'warning' | 'danger' | 'info' = 'info';

  // Subscriptions
  private pollSub?: Subscription;

  // Genuine MobileNetV2 AI Test Metrics
  aiPerformance = {
    architecture: 'MobileNetV2 Edge CNN (Transfer Learning)',
    inputDimensions: '160 × 160 × 3 (RGB float32)',
    modelFramework: 'TensorFlow Lite v2.1.0 (Edge Optimized)',
    overallTestAccuracy: 96.01,
    testLoss: 0.1451,
    avgInferenceLatencyMs: 18.4,
    totalDatasetImages: 6764,
    trainingImages: 5411,
    testingImages: 677,
    classes: [
      { name: 'Healthy Leaf', code: 'Healthy_Leaf', precision: 0.9306, recall: 0.9758, f1: 0.9527, testSamples: 165 },
      { name: 'Bacterial Leaf Spot', code: 'Bacterial_Spot', precision: 1.0000, recall: 0.9325, f1: 0.9651, testSamples: 163 },
      { name: 'Cercospora Leaf Spot', code: 'Cercospora_Leaf_Spot', precision: 0.9688, recall: 0.9789, f1: 0.9738, testSamples: 190 },
      { name: 'Chili Leaf Curl Virus', code: 'Curl_Virus', precision: 0.9438, recall: 0.9497, f1: 0.9467, testSamples: 159 }
    ]
  };

  constructor(public api: RoverApiService) {}

  ngOnInit(): void {
    this.apiEndpoint = this.api.getStoredApiUrl();
    this.refreshAllData();

    // 2.5 second live telemetry polling
    this.pollSub = interval(2500).subscribe(() => {
      this.fetchLiveStatus();
    });

    this.loadDiseaseEncyclopedia();

    // If accessing through cloud (Render), activate high-performance progressive frame streaming
    if (this.api.isCloudMode()) {
      this.useSingleFrameMode = true;
      this.streamTimestamp = Date.now();
    }
  }

  ngOnDestroy(): void {
    if (this.pollSub) {
      this.pollSub.unsubscribe();
    }
    if (this.frameTimer) {
      clearTimeout(this.frameTimer);
      this.frameTimer = null;
    }
  }

  get liveStreamUrl(): string {
    if (this.useSingleFrameMode || this.api.isCloudMode()) {
      return `${this.api.getLiveFrameUrl()}?t=${this.streamTimestamp}`;
    }
    return `${this.api.getLiveStreamUrl()}?t=${this.streamTimestamp}`;
  }

  setTab(tab: 'dashboard' | 'rover' | 'data' | 'analytics' | 'settings'): void {
    this.activeTab = tab;
    this.mobileMenuOpen = false;
    if (tab === 'data') {
      this.fetchHistory();
    } else if (tab === 'analytics') {
      this.fetchStatistics();
    }
  }

  toggleMobileMenu(): void {
    this.mobileMenuOpen = !this.mobileMenuOpen;
  }

  showAlert(msg: string, type: 'success' | 'warning' | 'danger' | 'info' = 'info', timeoutMs: number = 4000): void {
    this.alertMessage = msg;
    this.alertType = type;
    if (timeoutMs > 0) {
      setTimeout(() => {
        if (this.alertMessage === msg) {
          this.alertMessage = null;
        }
      }, timeoutMs);
    }
  }

  refreshAllData(): void {
    this.fetchLiveStatus();
    this.fetchHistory();
    this.fetchStatistics();
  }

  fetchLiveStatus(): void {
    this.api.getStatus().subscribe({
      next: (status) => {
        this.status = status;
        if (status.rover?.motor_status) {
          this.activeMovement = status.rover.motor_status.state || 'STOPPED';
          this.isMoving = (this.activeMovement !== 'STOPPED');
        }
        if (status.sprayer) {
          this.sprayerActive = status.sprayer.sprayer_active;
        }
        if (status.latest_detection) {
          this.latestDetection = status.latest_detection;
        }
      },
      error: () => {}
    });
  }

  fetchHistory(): void {
    this.api.getDetections(100, 0, this.dataFilterDisease, this.dataMinConfidence).subscribe({
      next: (res) => {
        if (res.success && res.detections) {
          this.historyList = res.detections;
          this.applyDataFilter();
          if (this.historyList.length > 0 && !this.latestDetection) {
            this.latestDetection = this.historyList[0];
          }
        }
      },
      error: () => {}
    });
  }

  applyDataFilter(): void {
    const q = this.dataSearchQuery.toLowerCase().trim();
    this.filteredHistoryList = this.historyList.filter(item => {
      const matchQuery = !q ||
        item.detection_id.toLowerCase().includes(q) ||
        item.predicted_disease.toLowerCase().includes(q) ||
        (item.image_filename && item.image_filename.toLowerCase().includes(q));

      const matchDisease = (this.dataFilterDisease === 'ALL') ||
        (item.predicted_disease.startsWith(this.dataFilterDisease));

      const matchConfidence = !this.dataMinConfidence || (item.confidence >= this.dataMinConfidence);

      return matchQuery && matchDisease && matchConfidence;
    });
  }

  fetchStatistics(): void {
    this.api.getStatistics().subscribe({
      next: (stats) => {
        this.statistics = stats;
      },
      error: () => {}
    });
  }

  getDistributionList(): { key: string; label: string; count: number }[] {
    if (!this.statistics || !this.statistics.disease_distribution) {
      return [];
    }
    const dist = this.statistics.disease_distribution;
    return Object.keys(dist).map(k => ({
      key: k,
      label: dist[k]?.label || k.replace(/_/g, ' '),
      count: Number(dist[k]?.count || 0)
    }));
  }

  loadDiseaseEncyclopedia(): void {
    this.api.getDiseaseInfo().subscribe({
      next: (res) => {
        if (res.success && res.diseases) {
          this.diseaseInfoList = res.diseases;
        }
      },
      error: () => {}
    });
  }

  // -----------------------------------------------------------
  // Camera Controls & Live Progressive Streaming
  // -----------------------------------------------------------
  onFrameLoaded(): void {
    this.streamError = false;
    this.streamLoaded = true;
    this.frameCount++;
    const now = Date.now();
    if (now - this.lastFpsCalcTime >= 1000) {
      this.liveFpsDisplay = Math.round((this.frameCount * 1000) / (now - this.lastFpsCalcTime));
      this.frameCount = 0;
      this.lastFpsCalcTime = now;
    }

    // Auto-advance next frame for cloud Render or progressive single frame fallback
    if (this.liveStreamActive && (this.useSingleFrameMode || this.api.isCloudMode())) {
      if (this.frameTimer) clearTimeout(this.frameTimer);
      this.frameTimer = setTimeout(() => {
        if (this.liveStreamActive) {
          this.streamTimestamp = Date.now();
        }
      }, 100); // 100ms pacing -> ultra-smooth ~10 FPS live streaming over Render
    }
  }

  toggleLiveStream(): void {
    this.liveStreamActive = !this.liveStreamActive;
    if (this.liveStreamActive) {
      this.streamError = false;
      this.streamTimestamp = Date.now();
      if (this.useSingleFrameMode || this.api.isCloudMode()) {
        if (this.frameTimer) clearTimeout(this.frameTimer);
        this.frameTimer = setTimeout(() => {
          this.streamTimestamp = Date.now();
        }, 100);
      }
    } else {
      if (this.frameTimer) {
        clearTimeout(this.frameTimer);
        this.frameTimer = null;
      }
    }
  }

  onStreamError(event: any): void {
    this.streamError = true;
    this.useSingleFrameMode = true;
    if (this.frameTimer) clearTimeout(this.frameTimer);
    this.frameTimer = setTimeout(() => {
      if (this.liveStreamActive) {
        this.streamError = false;
        this.streamTimestamp = Date.now();
      }
    }, 800);
  }

  @HostListener('window:keydown', ['$event'])
  handleKeyboardEvent(event: KeyboardEvent): void {
    if (this.activeTab !== 'rover') return;
    const target = event.target as HTMLElement;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')) {
      return;
    }

    const key = event.key.toUpperCase();
    if (key === 'W' || key === 'ARROWUP') {
      event.preventDefault();
      if (this.activeMovement !== 'FORWARD') this.sendRoverMove('FORWARD');
    } else if (key === 'S' || key === 'ARROWDOWN') {
      event.preventDefault();
      if (this.activeMovement !== 'BACKWARD') this.sendRoverMove('BACKWARD');
    } else if (key === 'A' || key === 'ARROWLEFT') {
      event.preventDefault();
      if (this.activeMovement !== 'LEFT') this.sendRoverMove('LEFT');
    } else if (key === 'D' || key === 'ARROWRIGHT') {
      event.preventDefault();
      if (this.activeMovement !== 'RIGHT') this.sendRoverMove('RIGHT');
    } else if (key === ' ' || key === 'X' || key === 'ESCAPE') {
      event.preventDefault();
      this.sendRoverMove('STOP');
    }
  }

  @HostListener('window:keyup', ['$event'])
  handleKeyboardUpEvent(event: KeyboardEvent): void {
    if (this.activeTab !== 'rover') return;
    const target = event.target as HTMLElement;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT')) {
      return;
    }
    const key = event.key.toUpperCase();
    if (['W', 'ARROWUP', 'S', 'ARROWDOWN', 'A', 'ARROWLEFT', 'D', 'ARROWRIGHT'].includes(key)) {
      event.preventDefault();
      this.sendRoverMove('STOP');
    }
  }

  triggerManualCapture(): void {
    this.showAlert('Capturing high-resolution camera snapshot...', 'info');
    this.api.captureLeaf(true).subscribe({
      next: (res) => {
        if (res.success) {
          this.showAlert('Snapshot captured & classified successfully!', 'success');
          if (res.detection) {
            this.latestDetection = res.detection;
          }
          this.fetchHistory();
          this.fetchLiveStatus();
        } else {
          this.showAlert(res.message || 'Capture could not be completed.', 'warning');
        }
      },
      error: (err) => {
        this.showAlert(`Camera capture error: ${err.message || 'Cannot contact camera'}`, 'danger');
      }
    });
  }

  toggleAutoDetect(): void {
    this.autoDetectEnabled = !this.autoDetectEnabled;
    this.api.setAutoDetect(this.autoDetectEnabled, this.autoInterval).subscribe({
      next: () => {
        this.showAlert(`Automated leaf detection set to ${this.autoDetectEnabled ? 'ON' : 'OFF'}`, 'info');
      },
      error: () => {}
    });
  }

  onFileUpload(event: any): void {
    const file = event.target.files?.[0];
    if (!file) return;

    this.showAlert(`Verifying botanical foliage and classifying ${file.name}...`, 'info');
    this.api.classifyUploadedLeaf(file).subscribe({
      next: (res) => {
        if (res.success && res.detection) {
          this.latestDetection = res.detection;
          if (res.detection.leaf_detected) {
            this.showAlert(`Diagnosis Complete: ${res.detection.predicted_disease} (${res.detection.confidence}%)`, 'success');
          } else {
            this.showAlert(`Non-Plant Object: ${res.detection.display_label}`, 'warning');
          }
          this.fetchHistory();
          this.fetchLiveStatus();
        } else {
          this.showAlert(`Classification failed: ${res.error}`, 'danger');
        }
      },
      error: (err) => {
        this.showAlert(`Upload failed: ${err.message}`, 'danger');
      }
    });
  }

  // -----------------------------------------------------------
  // Rover Motor Control
  // -----------------------------------------------------------
  sendRoverMove(direction: string): void {
    this.api.moveRover(direction, this.roverSpeed).subscribe({
      next: (res) => {
        if (res.success) {
          this.activeMovement = direction;
          this.isMoving = (direction !== 'STOP');
          this.showAlert(`Rover Command: ${direction} (${this.roverSpeed}%)`, 'info', 2000);
        } else {
          this.activeMovement = 'BLOCKED';
          this.showAlert(res.message || 'Movement blocked by obstacle safety lock!', 'warning');
        }
      },
      error: (err) => {
        this.showAlert(`Rover command error: ${err.message}`, 'danger');
      }
    });
  }

  emergencyStop(): void {
    this.api.emergencyStop().subscribe({
      next: () => {
        this.activeMovement = 'STOPPED';
        this.isMoving = false;
        this.showAlert('EMERGENCY HARD STOP ACTIVATED!', 'danger');
      },
      error: (err) => {
        this.showAlert(`Emergency stop error: ${err.message}`, 'danger');
      }
    });
  }

  setRoverMode(mode: string): void {
    this.api.setRoverMode(mode).subscribe(res => {
      if (res.success) {
        this.showAlert(`Rover mode switched to: ${mode}`, 'success');
        this.fetchLiveStatus();
      }
    });
  }

  // -----------------------------------------------------------
  // Manual-Only Sprayer System Controls
  // -----------------------------------------------------------
  turnSprayerOn(duration?: number): void {
    const dur = duration !== undefined ? duration : this.sprayDuration;
    this.sprayerActive = true;
    this.showAlert(`💦 Turning Sprayer ON (Manual Operator Mode - ${dur > 0 ? dur + 's' : 'Continuous'})...`, 'warning', 3000);
    this.api.activateSprayer(dur, true, { mode: 'MANUAL', target: this.sprayTargetDisease || 'Manual Operation' }).subscribe({
      next: (res) => {
        if (res.success) {
          this.sprayerActive = true;
          this.showAlert(`✅ Sprayer Pump is now ACTIVE (ON)!`, 'success', 3000);
          if (dur > 0) {
            setTimeout(() => {
              this.fetchLiveStatus();
            }, (dur * 1000) + 300);
          }
        } else {
          this.sprayerActive = false;
          this.showAlert(`Sprayer error: ${res.message || 'Cannot activate pump'}`, 'danger');
        }
      },
      error: (err) => {
        this.sprayerActive = false;
        this.showAlert(`Sprayer communication error: ${err.message}`, 'danger');
      }
    });
  }

  turnSprayerOff(): void {
    this.sprayerActive = false;
    this.showAlert('🛑 Turning Sprayer OFF...', 'info', 2000);
    this.api.stopSprayer().subscribe({
      next: (res) => {
        this.sprayerActive = false;
        this.showAlert('✅ Sprayer Pump is now OFF (Standby).', 'success', 2500);
        this.fetchLiveStatus();
      },
      error: (err) => {
        this.showAlert(`Sprayer stop error: ${err.message}`, 'danger');
      }
    });
  }

  toggleManualSprayer(): void {
    if (this.sprayerActive) {
      this.turnSprayerOff();
    } else {
      this.turnSprayerOn();
    }
  }

  promptSpray(diseaseName?: string): void {
    this.sprayTargetDisease = diseaseName || (this.latestDetection ? this.latestDetection.predicted_disease : 'Manual Operation');
    // Direct manual activation without blocking modals
    this.turnSprayerOn(this.sprayDuration);
  }

  confirmSprayer(): void {
    this.showSprayConfirmModal = false;
    this.turnSprayerOn(this.sprayDuration);
  }

  stopSprayer(): void {
    this.turnSprayerOff();
  }

  // -----------------------------------------------------------
  // Settings Management
  // -----------------------------------------------------------
  applySettings(): void {
    this.api.setApiUrl(this.apiEndpoint);
    this.showAlert(`API endpoint updated to ${this.apiEndpoint}. Re-synchronizing...`, 'success');
    this.refreshAllData();
  }

  setApiPreset(preset: string): void {
    this.selectedCloudPreset = preset;
    if (preset === 'render') {
      this.apiEndpoint = 'https://precision-agriculture-smart-rover-5xxg.onrender.com';
    } else if (preset === 'local') {
      this.apiEndpoint = 'http://localhost:5001';
    }
  }

  triggerCloudSync(): void {
    this.showAlert('Triggering cloud telemetry synchronization...', 'info');
    this.api.triggerSync().subscribe(res => {
      if (res.success) {
        this.showAlert(`Sync Complete! Synced: ${res.synced_detections || 0} detections.`, 'success');
        this.refreshAllData();
      } else {
        this.showAlert(`Sync notice: ${res.message || res.error}`, 'warning');
      }
    });
  }

  downloadCsv(): void {
    window.open(this.api.exportCsvUrl(), '_blank');
  }

  getDiseaseBadgeClass(disease?: string): string {
    if (!disease) return 'badge-neutral';
    if (disease.includes('Healthy')) return 'badge-healthy';
    if (disease.includes('Bacterial')) return 'badge-bacterial';
    if (disease.includes('Cercospora')) return 'badge-cercospora';
    if (disease.includes('Curl')) return 'badge-curl';
    return 'badge-neutral';
  }
}
