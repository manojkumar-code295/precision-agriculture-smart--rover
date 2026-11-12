import { Injectable } from '@angular/core';
import { HttpClient, HttpHeaders, HttpParams } from '@angular/common/http';
import { Observable, BehaviorSubject, of } from 'rxjs';
import { catchError, tap } from 'rxjs/operators';

export interface SystemStatus {
  device_id: string;
  device_name: string;
  pi_online: boolean;
  last_seen?: string;
  camera: {
    camera_connected: boolean;
    camera_type: string;
    last_capture: string | null;
    error_message: string | null;
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
    sensors: {
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
  predicted_disease: string;
  disease_code?: string;
  confidence: number;
  is_low_confidence: boolean;
  recommendation: string;
  sprayer_status: string;
  synced_at?: string;
}

@Injectable({
  providedIn: 'root'
})
export class RoverApiService {
  private defaultApiUrl = 'http://localhost:5000';
  private apiUrlKey = 'rover_api_endpoint';
  public baseUrl$ = new BehaviorSubject<string>(this.getStoredApiUrl());
  public liveStatus$ = new BehaviorSubject<SystemStatus | null>(null);

  constructor(private http: HttpClient) {}

  public getStoredApiUrl(): string {
    if (typeof window !== 'undefined' && window.localStorage) {
      const stored = window.localStorage.getItem(this.apiUrlKey);
      if (stored) return stored;
    }
    return this.defaultApiUrl;
  }

  public setApiUrl(url: string): void {
    const cleanUrl = url.trim().replace(/\/$/, '');
    if (typeof window !== 'undefined' && window.localStorage) {
      window.localStorage.setItem(this.apiUrlKey, cleanUrl);
    }
    this.baseUrl$.next(cleanUrl);
  }

  private get url(): string {
    return this.baseUrl$.getValue();
  }

  // 1. Health & Status
  getStatus(): Observable<SystemStatus> {
    return this.http.get<SystemStatus>(`${this.url}/api/status`).pipe(
      tap(status => this.liveStatus$.next(status)),
      catchError(err => {
        console.warn('Status request failed:', err);
        return of({
          device_id: 'offline-rover',
          device_name: 'Smart Rover (Reconnecting)',
          pi_online: false,
          camera: { camera_connected: false, camera_type: 'Offline', last_capture: null, error_message: 'Cannot reach rover API' },
          rover: { mode: 'MANUAL', is_moving: false, motor_status: { state: 'STOPPED', speed: 0, hardware_enabled: false }, obstacle_status: { obstacle_detected: false, distance_cm: null, hazard_level: 'OFFLINE' } },
          sensors: { sensors: {} },
          sprayer: { sprayer_active: false, hardware_enabled: false, last_activated: null },
          timestamp: new Date().toISOString()
        } as SystemStatus);
      })
    );
  }

  // 2. Camera & Detection
  captureLeaf(autoClassify: boolean = true): Observable<any> {
    return this.http.post<any>(`${this.url}/api/camera/capture?classify=${autoClassify}`, {});
  }

  classifyUploadedLeaf(file: File): Observable<any> {
    const formData = new FormData();
    formData.append('image', file, file.name);
    return this.http.post<any>(`${this.url}/api/ai/classify`, formData);
  }

  getLatestDetection(): Observable<any> {
    return this.http.get<any>(`${this.url}/api/detections/latest`);
  }

  getDetections(limit: number = 50, offset: number = 0, disease?: string, minConfidence?: number): Observable<any> {
    let params = new HttpParams().set('limit', limit).set('offset', offset);
    if (disease) params = params.set('disease', disease);
    if (minConfidence !== undefined) params = params.set('min_confidence', minConfidence);
    return this.http.get<any>(`${this.url}/api/detections`, { params });
  }

  // 3. Rover Control
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

  // 4. Sprayer Control
  activateSprayer(duration: number, confirmed: boolean, diseaseContext?: any): Observable<any> {
    return this.http.post<any>(`${this.url}/api/sprayer/activate`, {
      duration,
      confirmed,
      disease_context: diseaseContext
    });
  }

  stopSprayer(): Observable<any> {
    return this.http.post<any>(`${this.url}/api/sprayer/stop`, {});
  }

  // 5. Sensors & Telemetry
  getSensors(): Observable<any> {
    return this.http.get<any>(`${this.url}/api/sensors`);
  }

  // 6. Statistics & Reports
  getStatistics(): Observable<any> {
    return this.http.get<any>(`${this.url}/api/statistics`);
  }

  getDiseaseInfo(): Observable<any> {
    return this.http.get<any>(`${this.url}/api/disease-info`);
  }

  exportCsvUrl(): string {
    return `${this.url}/api/reports/export.csv`;
  }

  // 7. Cloud Sync
  triggerSync(): Observable<any> {
    return this.http.post<any>(`${this.url}/api/sync/trigger`, {});
  }
}
