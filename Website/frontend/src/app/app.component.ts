import { Component, OnInit, OnDestroy } from '@angular/core';
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
  title = 'Plant Disease Detection AI - Precision Agriculture Smart Rover';
  activeTab: string = 'dashboard';
  mobileMenuOpen: boolean = false;

  // Live Subsystem Status
  status: SystemStatus | null = null;
  latestDetection: DetectionRecord | null = null;
  historyList: DetectionRecord[] = [];
  statistics: any = null;
  diseaseInfoList: any[] = [];
  telemetry: any = null;

  // Rover Movement
  roverSpeed: number = 75;
  activeMovement: string = 'STOPPED';
  isMoving: boolean = false;
  obstacleStatus: any = null;

  // Live Camera Stream & Detection
  liveStreamActive: boolean = true;
  streamError: boolean = false;
  autoDetectEnabled: boolean = true;
  autoInterval: number = 3.0;
  streamTimestamp: number = Date.now();

  get liveStreamUrl(): string {
    return `${this.api.getLiveStreamUrl()}?t=${this.streamTimestamp}`;
  }

  toggleLiveStream(): void {
    this.liveStreamActive = !this.liveStreamActive;
    if (this.liveStreamActive) {
      this.streamError = false;
      this.streamTimestamp = Date.now();
    }
  }

  onStreamError(event: any): void {
    console.warn('Live stream disconnected or awaiting frames:', event);
    this.streamError = true;
    // Automatically retry connecting after 2.0s
    setTimeout(() => {
      if (this.liveStreamActive) {
        this.streamError = false;
        this.streamTimestamp = Date.now();
      }
    }, 2000);
  }

  toggleAutoDetect(): void {
    this.autoDetectEnabled = !this.autoDetectEnabled;
    this.api.setAutoDetect(this.autoDetectEnabled, this.autoInterval).subscribe({
      next: (res) => {
        this.showAlert(`Auto leaf detection set to ${this.autoDetectEnabled ? 'ON' : 'OFF'}`, 'info');
      },
      error: () => {}
    });
  }

  // Sprayer State
  sprayDuration: number = 3.0;
  sprayerActive: boolean = false;
  showSprayConfirmModal: boolean = false;
  sprayTargetDisease: string = '';

  // History Filter
  historyFilterDisease: string = '';
  historyMinConfidence: number = 0;

  // Settings
  apiEndpoint: string = '';
  confidenceThreshold: number = 70;
  obstacleDistanceThreshold: number = 25;
  selectedCloudPreset: string = 'local';

  // UI Alerts
  alertMessage: string | null = null;
  alertType: 'success' | 'warning' | 'danger' | 'info' = 'info';

  // Subscriptions
  private pollSub?: Subscription;
  private frameSub?: Subscription;

  // Dataset Static Metadata (from Stage 2 Verified Reports)
  datasetInfo = {
    totalImages: 6764,
    categories: [
      { name: 'Bacterial Spot', folder: 'Bacterial_Spot', count: 1629, precision: 1.0000, recall: 0.9325, f1: 0.9651 },
      { name: 'Cercospora Leaf Spot', folder: 'Cercospora_Leaf_Spot', count: 1898, precision: 0.9688, recall: 0.9789, f1: 0.9738 },
      { name: 'Curl Virus', folder: 'Curl_Virus', count: 1590, precision: 0.9437, recall: 0.9497, f1: 0.9467 },
      { name: 'Healthy Leaf', folder: 'Healthy_Leaf', count: 1647, precision: 0.9306, recall: 0.9758, f1: 0.9527 }
    ],
    architecture: 'MobileNetV2 CNN (Transfer Learning)',
    inputShape: '160 x 160 x 3 (RGB float32)',
    testAccuracy: 96.01,
    testLoss: 0.1451,
    tfliteSizeMb: 8.49,
    kerasSizeMb: 20.78
  };

  constructor(public api: RoverApiService) {}

  ngOnInit(): void {
    this.apiEndpoint = this.api.getStoredApiUrl();
    this.refreshAllData();

    // Start 3-second live telemetry poll
    this.pollSub = interval(3000).subscribe(() => {
      this.fetchLiveStatus();
    });

    // High-speed progressive frame refresh for cloud live frame mode (2 FPS)
    this.frameSub = interval(500).subscribe(() => {
      if (this.liveStreamActive && this.api.isCloudMode() && !this.streamError) {
        this.streamTimestamp = Date.now();
      }
    });

    this.loadDiseaseEncyclopedia();
  }

  ngOnDestroy(): void {
    if (this.pollSub) {
      this.pollSub.unsubscribe();
    }
    if (this.frameSub) {
      this.frameSub.unsubscribe();
    }
  }

  setTab(tabName: string): void {
    this.activeTab = tabName;
    this.mobileMenuOpen = false;
    if (tabName === 'history') {
      this.fetchHistory();
    } else if (tabName === 'analytics') {
      this.fetchStatistics();
    }
  }

  toggleMobileMenu(): void {
    this.mobileMenuOpen = !this.mobileMenuOpen;
  }

  showAlert(msg: string, type: 'success' | 'warning' | 'danger' | 'info' = 'info'): void {
    this.alertMessage = msg;
    this.alertType = type;
    setTimeout(() => {
      if (this.alertMessage === msg) {
        this.alertMessage = null;
      }
    }, 5000);
  }

  // -----------------------------------------------------------
  // Polling & Data Retrieval
  // -----------------------------------------------------------
  refreshAllData(): void {
    this.fetchLiveStatus();
    this.fetchHistory();
    this.fetchStatistics();
    this.fetchSensors();
  }

  fetchLiveStatus(): void {
    this.api.getStatus().subscribe(res => {
      this.status = res;
      this.obstacleStatus = res?.rover?.obstacle_status;
      this.sprayerActive = !!res?.sprayer?.sprayer_active;
      if (res?.latest_detection) {
        this.latestDetection = res.latest_detection;
      }
      if (this.streamError && res?.pi_online) {
        this.streamError = false;
      }
    });
  }

  fetchHistory(): void {
    const minConf = this.historyMinConfidence > 0 ? this.historyMinConfidence : undefined;
    const disease = this.historyFilterDisease || undefined;
    this.api.getDetections(50, 0, disease, minConf).subscribe(res => {
      if (res?.detections) {
        this.historyList = res.detections;
      }
    });
  }

  fetchStatistics(): void {
    this.api.getStatistics().subscribe(res => {
      if (res) {
        this.statistics = res;
      }
    });
  }

  fetchSensors(): void {
    this.api.getSensors().subscribe(res => {
      if (res) {
        this.telemetry = res;
      }
    });
  }

  loadDiseaseEncyclopedia(): void {
    this.api.getDiseaseInfo().subscribe(res => {
      if (res?.diseases) {
        this.diseaseInfoList = res.diseases;
      }
    });
  }

  // -----------------------------------------------------------
  // Camera & Detection Controls
  // -----------------------------------------------------------
  captureImage(): void {
    this.showAlert('Capturing leaf image from Raspberry Pi camera...', 'info');
    this.api.captureLeaf(true).subscribe({
      next: (res) => {
        if (res.success) {
          if (res.detection) {
            this.latestDetection = res.detection;
            this.showAlert(`Leaf Classified: ${res.detection.predicted_disease} (${res.detection.confidence}%)`, 'success');
            this.fetchHistory();
          } else {
            this.showAlert('Image captured successfully.', 'success');
          }
        } else {
          this.showAlert(`Capture failed: ${res.error}`, 'danger');
        }
      },
      error: (err) => {
        this.showAlert(`Camera capture error: ${err.message || 'Cannot contact camera'}`, 'danger');
      }
    });
  }

  onFileUpload(event: any): void {
    const file = event.target.files?.[0];
    if (!file) return;

    this.showAlert(`Uploading ${file.name} for AI classification...`, 'info');
    this.api.classifyUploadedLeaf(file).subscribe({
      next: (res) => {
        if (res.success && res.detection) {
          this.latestDetection = res.detection;
          this.showAlert(`Diagnosis Complete: ${res.detection.predicted_disease} (${res.detection.confidence}%)`, 'success');
          this.fetchHistory();
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
          this.showAlert(`Rover command: ${direction} (${this.roverSpeed}%)`, 'info');
        } else {
          this.activeMovement = 'BLOCKED';
          this.showAlert(res.message || 'Movement blocked!', 'warning');
        }
      },
      error: (err) => {
        this.showAlert(`Rover communication error: ${err.message}`, 'danger');
      }
    });
  }

  emergencyStop(): void {
    this.api.emergencyStop().subscribe({
      next: () => {
        this.activeMovement = 'EMERGENCY_STOP';
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
  // Sprayer System Controls
  // -----------------------------------------------------------
  promptSpray(diseaseName?: string): void {
    this.sprayTargetDisease = diseaseName || (this.latestDetection ? this.latestDetection.predicted_disease : 'Manual Operation');
    this.showSprayConfirmModal = true;
  }

  confirmSprayer(): void {
    this.showSprayConfirmModal = false;
    this.showAlert(`Actuating chemical sprayer for ${this.sprayDuration}s...`, 'warning');
    this.api.activateSprayer(this.sprayDuration, true, { disease: this.sprayTargetDisease }).subscribe({
      next: (res) => {
        if (res.success) {
          this.sprayerActive = true;
          this.showAlert(`Sprayer activated for ${this.sprayDuration}s.`, 'success');
          setTimeout(() => {
            this.sprayerActive = false;
          }, this.sprayDuration * 1000);
        } else {
          this.showAlert(`Sprayer rejected: ${res.message}`, 'danger');
        }
      },
      error: (err) => {
        this.showAlert(`Sprayer error: ${err.message}`, 'danger');
      }
    });
  }

  stopSprayer(): void {
    this.api.stopSprayer().subscribe(() => {
      this.sprayerActive = false;
      this.showAlert('Sprayer deactivated immediately.', 'info');
    });
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
    if (preset === 'local') {
      this.apiEndpoint = 'http://localhost:5000';
    } else if (preset === 'render') {
      this.apiEndpoint = 'https://precision-agriculture-smart-rover.onrender.com';
    }
  }

  triggerCloudSync(): void {
    this.showAlert('Triggering cloud synchronization...', 'info');
    this.api.triggerSync().subscribe(res => {
      if (res.success) {
        this.showAlert(`Sync Complete! Synced: ${res.synced_detections || 0} detections.`, 'success');
      } else {
        this.showAlert(`Sync notice: ${res.message || res.error}`, 'warning');
      }
    });
  }

  downloadCsv(): void {
    window.open(this.api.exportCsvUrl(), '_blank');
  }

  getImageUrl(filename?: string): string {
    if (!filename) return 'assets/placeholder-leaf.png';
    return `${this.api.baseUrl$.getValue()}/captures/${filename}`;
  }
}
