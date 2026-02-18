/**
 * Authentication Service
 * Following Single Responsibility Principle - handles all auth-related API calls
 * Follows Dependency Inversion - depends on interfaces, not concrete implementations
 */

import { API_CONFIG, HTTP_STATUS } from '@/config/api.config';
import type { SignupData, LoginData, AuthResponse, ApiError } from '@/types/auth.types';
import { AuthServiceError } from '@/errors/auth.error';

/**
 * HTTP client wrapper with error handling
 */
class HttpClient {
  private baseURL: string;
  private timeout: number;

  constructor(baseURL: string, timeout: number) {
    this.baseURL = baseURL;
    this.timeout = timeout;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(`${this.baseURL}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Handle non-OK responses
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({
          message: 'An unexpected error occurred',
        }));

        throw new AuthServiceError(
          errorData.message || `HTTP ${response.status}`,
          response.status,
          errorData.field
        );
      }

      return await response.json();
    } catch (error) {
      clearTimeout(timeoutId);

      if (error instanceof AuthServiceError) {
        throw error;
      }

      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new AuthServiceError('Request timeout - please try again');
        }
        throw new AuthServiceError(error.message);
      }

      throw new AuthServiceError('An unexpected error occurred');
    }
  }

  async post<T>(endpoint: string, body: unknown): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'POST',
      body: JSON.stringify(body),
    });
  }

  async get<T>(endpoint: string): Promise<T> {
    return this.request<T>(endpoint, {
      method: 'GET',
    });
  }
}

/**
 * Authentication Service Class
 * Provides methods for user authentication operations
 */
class AuthService {
  private client: HttpClient;

  constructor() {
    this.client = new HttpClient(API_CONFIG.BASE_URL, API_CONFIG.TIMEOUT);
  }

  /**
   * Register a new user
   */
  async signup(data: Omit<SignupData, 'confirmPassword'>): Promise<AuthResponse> {
    try {
      const response = await this.client.post<AuthResponse>(
        API_CONFIG.ENDPOINTS.SIGNUP,
        {
          email: data.email.toLowerCase().trim(),
          password: data.password,
        }
      );

      // Store token if provided
      if (response.token) {
        this.setAuthToken(response.token);
      }

      return response;
    } catch (error) {
      if (error instanceof AuthServiceError) {
        throw error;
      }
      throw new AuthServiceError('Signup failed - please try again');
    }
  }

  /**
   * Login an existing user
   */
  async login(data: LoginData): Promise<AuthResponse> {
    try {
      const response = await this.client.post<AuthResponse>(
        API_CONFIG.ENDPOINTS.LOGIN,
        {
          email: data.email.toLowerCase().trim(),
          password: data.password,
        }
      );

      // Store token if provided
      if (response.token) {
        this.setAuthToken(response.token);
      }

      return response;
    } catch (error) {
      if (error instanceof AuthServiceError) {
        throw error;
      }
      throw new AuthServiceError('Login failed - please try again');
    }
  }

  /**
   * Logout current user
   */
  async logout(): Promise<void> {
    this.removeAuthToken();
  }

  /**
   * Get current user info
   */
  async getCurrentUser(): Promise<AuthResponse> {
    const token = this.getAuthToken();
    if (!token) {
      throw new AuthServiceError('No authentication token found', HTTP_STATUS.UNAUTHORIZED);
    }

    return this.client.get<AuthResponse>(API_CONFIG.ENDPOINTS.ME);
  }

  /**
   * Store authentication token
   */
  private setAuthToken(token: string): void {
    if (typeof window !== 'undefined') {
      localStorage.setItem('authToken', token);
    }
  }

  /**
   * Get authentication token
   */
  private getAuthToken(): string | null {
    if (typeof window !== 'undefined') {
      return localStorage.getItem('authToken');
    }
    return null;
  }

  /**
   * Remove authentication token
   */
  private removeAuthToken(): void {
    if (typeof window !== 'undefined') {
      localStorage.removeItem('authToken');
    }
  }
}

// Export singleton instance
export const authService = new AuthService();
export { AuthServiceError };