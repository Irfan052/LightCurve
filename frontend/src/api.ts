import axios from "axios";

const API_BASE_URL = "http://localhost:8000/api";

export interface AnalysisResult {
  type?: "lightcurve";
  status?: string;
  message?: string;
  target?: string;
  target_name?: string;
  is_mock?: boolean;
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  parameters: {
    transit_depth_percent?: number;
    transit_duration_hours?: number;
    planet_radius_earth?: number;
    period?: number;
    snr?: number;
    epoch?: number;
    semi_major_axis_au?: number;
    [key: string]: unknown;
  };
  features: Record<string, number>;
  feature_importance_summary: {
    top_random_forest?: Array<{ feature: string; importance: number }>;
    [key: string]: unknown;
  };

  data: {
    time: number[];
    flux: number[];

    time_clean: number[];
    flux_clean: number[];

    flat_flux: number[];
    trend_flux: number[];

    folded_phase: number[];
    folded_flux: number[];

    bin_centers: number[];
    bin_flux: number[];

    bls_periods: number[];
    bls_powers: number[];
  };
  metadata?: {
    mission: string;
    author: string;
    sector: string;
    cadence: string;
    collection: string;
    observation_id: string;
    product_type: string;
    provenance: string;
    filename: string;
  };
}

export interface CatalogResult {
  type: "catalog";
  catalog_type: "tic_catalog" | "gaia_catalog" | "mixed_catalog";
  columns: string[];
  objects: Record<string, any>[];
  total_rows?: number;
  preview_rows?: number;
  valid_targets?: number;
  skipped_rows?: number;
}

export const checkHealth = async () => {
  try {
    const response = await axios.get(`${API_BASE_URL}/health`);
    return response.data;
  } catch (error) {
    console.error("Backend health check failed:", error);
    return null;
  }
};

export async function analyzeTarget(
  targetId: string
): Promise<AnalysisResult> {
  const response = await axios.post(`${API_BASE_URL}/analyze`, {
    target_id: targetId,
  }, { timeout: 90000 });
  return response.data;
}

export async function uploadAndAnalyze(
  file: File
): Promise<AnalysisResult | CatalogResult> {
  const formData = new FormData();
  formData.append("file", file);

  const response = await axios.post(
    `${API_BASE_URL}/upload`,
    formData,
    {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    }
  );
  return response.data;
}

export async function generateReport(data: AnalysisResult): Promise<Blob> {
  const response = await axios.post(`${API_BASE_URL}/report`, data, {
    responseType: "blob",
  });
  return response.data;
}