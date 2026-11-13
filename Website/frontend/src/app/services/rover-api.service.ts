import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable, BehaviorSubject, of } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';

export interface SystemStatus {
  device_id: string;
  device_name: string;
  pi_online: boolean;
  pi_status: 'ONLINE' | 'OFFLINE';
  camera_status: 'STREAMING' | 'INTERRUPTED';
  ai_status: 'RUNNING' | 'ERROR';
  fps?: number;
  battery?: {
    percentage: number;
    voltage: number;
    status: string;
  };
  last_seen?: string;
  camera: {
    camera_connected: boolean;
    status?: 'STREAMING' | 'INTERRUPTED';
    camera_type: string;
    live_fps?: number;
    resolution?: string;
    last_capture: string | null;
    error_message: string | null;
  };
  ai?: {
    status: 'RUNNING' | 'ERROR';
    model_version: string;
    architecture: string;
    input_shape: string;
    classes: string[];
  };
  rover: {
    mode: string;
    is_moving: boolean;
    motor_status: {
      state: string;
      speed: number;
      hardware_enabled: boolean;
    };
    obstacle_status: {
      obstacle_detected: boolean;
      distance_cm: number | null;
      hazard_level: string;
      safe_to_move_forward?: boolean;
      message?: string;
    };
  };
  sensors: {
    timestamp?: string;
    sensors?: {
      [key: string]: {
        name: string;
        installed: boolean;
        connected: boolean;
        reading: any;
        unit: any;
        status: string;
        error?: string | null;
      };
    };
    [key: string]: any;
  };
  sprayer: {
    sprayer_active: boolean;
    hardware_enabled: boolean;
    last_activated: string | null;
    default_duration_sec?: number;
  };
  sync?: {
    online: boolean;
    cloud_url: string;
    last_sync: string | null;
  };
  latest_detection?: any;
  statistics?: any;
  total_detections?: number;
  healthy_count?: number;
  diseased_count?: number;
  timestamp: string;
}

export interface DetectionRecord {
  id?: number;
  detection_id: string;
  timestamp: string;
  image_filename: string;
  image_url?: string;
  leaf_detected?: boolean;
  display_label?: string;
  predicted_disease: string;
  disease_code?: string;
  confidence: number;
  is_low_confidence: boolean;
  recommendation: string;
  sprayer_status: string;
  synced_at?: string;
  device_id?: string;
}

@Injectable({
  providedIn: 'root'
})
export class RoverApiService {
  private defaultCloudUrl = 'https://precision-agriculture-smart-rover-5xxg.onrender.com';
  private apiUrlKey = 'rover_api_endpoint';
  public baseUrl$ = new BehaviorSubject<string>(this.getStoredApiUrl());
  public liveStatus$ = new BehaviorSubject<SystemStatus | null>(null);

  constructor(private http: HttpClient) {}

  public getStoredApiUrl(): string {
    if (typeof window !== 'undefined' && window.localStorage) {
      const stored = window.localStorage.getItem(this.apiUrlKey);
      if (stored) return stored;

      if (window.location && window.location.hostname) {
        const host = window.location.hostname;
        const port = window.location.port;

        // If running directly on Render or behind web proxy
        if (host.includes('onrender.com')) {
          return window.location.origin;
        }

        // If accessing local Pi or desktop
        if (host === 'localhost' || host === '127.0.0.1') {
          if (port === '5001' || port === '5000') {
            return window.location.origin;
          }
          return 'http://localhost:5001';
        }

        // LAN access (e.g. agrirover.local or 192.168.x.x)
        return `http://${host}:5001`;
      }
    }
    return this.defaultCloudUrl;
  }

  public setApiUrl(url: string): void {
    const cleanUrl = url.trim().replace(/\/$/, '');
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(this.apiUrlKey, cleanUrl);
    }
    this.baseUrl$.next(cleanUrl);
  }

  public get url(): string {
    return this.baseUrl$.getValue();
  }

  public isCloudMode(): boolean {
    return this.url.includes('onrender.com');
  }

  // 1. Health & Live Status
  getStatus(): Observable<SystemStatus> {
    return this.http.get<SystemStatus>(`${this.url}/api/status`).pipe(
      tap(status => {
        // Enforce consistent boolean status
        if (!status.pi_status) {
          status.pi_status = status.pi_online ? 'ONLINE' : 'OFFLINE';
        }
        if (!status.camera_status) {
          const isCam = status.camera?.camera_connected || status.camera?.status === 'STREAMING';
          status.camera_status = isCam ? 'STREAMING' : 'INTERRUPTED';
        }
        if (!status.ai_status) {
          status.ai_status = status.ai?.status === 'RUNNING' ? 'RUNNING' : 'ERROR';
        }
        this.liveStatus$.next(status);
      }),
      catchError(err => {
        console.warn('Status request failed:', err);
        const fallback: SystemStatus = {
          device_id: 'agrirover-pi4-01',
          device_name: 'Precision Smart Rover Pi4',
          pi_online: false,
          pi_status: 'OFFLINE',
          camera_status: 'INTERRUPTED',
          ai_status: 'ERROR',
          fps: 0,
          battery: { percentage: 0, voltage: 0, status: 'DISCONNECTED' },
          camera: {
            camera_connected: false,
            status: 'INTERRUPTED',
            camera_type: 'Offline',
            live_fps: 0,
            resolution: '640x480',
            last_capture: null,
            error_message: 'Cannot contact rover API'
          },
          ai: {
            status: 'ERROR',
            model_version: '2.1.0-Edge',
            architecture: 'MobileNetV2 CNN',
            input_shape: '160x160x3',
            classes: ['Bacterial_Spot', 'Cercospora_Leaf_Spot', 'Curl_Virus', 'Healthy_Leaf']
          },
          rover: {
            mode: 'MANUAL',
            is_moving: false,
            motor_status: { state: 'STOPPED', speed: 0, hardware_enabled: false },
            obstacle_status: { obstacle_detected: false, distance_cm: null, hazard_level: 'OFFLINE' }
          },
          sensors: { sensors: {} },
          sprayer: { sprayer_active: false, hardware_enabled: false, last_activated: null },
          timestamp: new Date().toISOString()
        };
        this.liveStatus$.next(fallback);
        return of(fallback);
      })
    );
  }

  // 2. Camera Streams & Frames
  getLiveStreamUrl(): string {
    return `${this.url}/api/camera/stream`;
  }

  getLiveFrameUrl(): string {
    return `${this.url}/api/camera/live_frame`;
  }

  captureLeaf(autoClassify: boolean = true): Observable<any> {
    return this.http.post<any>(`${this.url}/api/camera/capture?classify=${autoClassify}`, {});
  }

  setAutoDetect(enabled: boolean, interval: number = 3.0): Observable<any> {
    return this.http.post<any>(`${this.url}/api/camera/auto-detect`, { enabled, interval });
  }

  classifyUploadedLeaf(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('image', file, file.name);
    return this.http.post<any>(`${this.url}/api/ai/classify`, formData);
  }

  // 3. Detections & Data
  getLatestDetection(): Observable<any> {
    return this.http.get<any>(`${this.url}/api/detections/latest`);
  }

  getDetections(limit: number = 50, offset: number = 0, disease?: string, minConfidence?: number): Observable<any> {
    let params = new HttpParams().set('limit', limit).set('offset', offset);
    if (disease && disease !== 'ALL') params = params.set('disease', disease);
    if (minConfidence !== undefined && minConfidence > 0) params = params.set('min_confidence', minConfidence);
    return this.http.get<any>(`${this.url}/api/detections`, { params });
  }

  // 4. Rover Controls
  moveRover(command: string, speed: number = 75): Observable<any> {
    return this.http.post<any>(`${this.url}/api/rover/move`, { command, speed });
  }

  stopRover(): Observable<any> {
    return this.http.post<any>(`${this.url}/api/rover/stop`, {});
  }

  emergencyStop(): Observable<any> {
    return this.http.post<any>(`${this.url}/api/rover/emergency-stop`, {});
  }

  setRoverMode(mode: string): Observable<any> {
    return this.http.post<any>(`${this.url}/api/rover/mode`, { mode });
  }

  // 5. Sprayer Controls
  activateSprayer(duration: number = 3.0, confirmed: boolean = true, diseaseContext?: any): Observable<any> {
    return this.http.post<any>(`${this.url}/api/sprayer/activate`, {
      action: 'ON',
      duration,
      confirmed: true,
      disease_context: diseaseContext
    });
  }

  stopSprayer(): Observable<any> {
    return this.http.post<any>(`${this.url}/api/sprayer/stop`, {
      action: 'OFF'
    });
  }

  toggleSprayer(duration: number = 3.0): Observable<any> {
    return this.http.post<any>(`${this.url}/api/sprayer/toggle`, { duration });
  }

  // 6. Statistics, Disease Info, CSV Export
  getStatistics(): Observable<any> {
    return this.http.get<any>(`${this.url}/api/statistics`);
  }

  getDiseaseInfo(): Observable<any> {
    return this.http.get<any>(`${this.url}/api/disease-info`);
  }

  exportCsvUrl(): string {
    return `${this.url}/api/reports/export.csv`;
  }

  triggerSync(): Observable<any> {
    return this.http.post<any>(`${this.url}/api/sync/trigger`, {});
  }
}
