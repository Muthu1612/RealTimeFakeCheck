/**
 * Authentication Service
 * Following Single Responsibility Principle - handles all auth-related API calls
 * Follows Dependency Inversion - depends on interfaces, not concrete implementations
 */

import { API_CONFIG, HTTP_STATUS } from '@/config/api.config';
import type { SignupData, LoginData, AuthResponse, ApiError } from '@/types/auth.types';
import { AuthServiceError } from '@/errors/auth.error';

/**
 * Token provider interface for DIP compliance
 */
interface TokenProvider {
  getToken(): Promise<string | null>;
  clearToken(): void;
}

/**
 * HTTP client wrapper with error handling
 * Supports automatic token refresh and authorization headers
 */
class HttpClient {
  private baseURL: string;
  private timeout: number;
  private tokenProvider: TokenProvider | null;
  private refreshCallback: (() => Promise<void>) | null = null;

  constructor(
    baseURL: string,
    timeout: number,
    tokenProvider: TokenProvider | null = null
  ) {
    this.baseURL = baseURL;
    this.timeout = timeout;
    this.tokenProvider = tokenProvider;
  }

  /**
   * Set callback for token refresh
   */
  setRefreshCallback(callback: () => Promise<void>): void {
    this.refreshCallback = callback;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {},
    retry: boolean = true
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      // Get token from provider if available
      const token = this.tokenProvider
        ? await this.tokenProvider.getToken()
        : null;

      const response = await fetch(`${this.baseURL}${endpoint}`, {
        ...options,
        headers: {
          'Content-Type': 'application/json',
          ...(token && { Authorization: `Bearer ${token}` }),
          ...options.headers,
        },
        credentials: 'include', // Send HTTP-only cookies automatically
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      // Handle 401 - Unauthorized (token expired)
      if (response.status === HTTP_STATUS.UNAUTHORIZED && retry && this.refreshCallback) {
        try {
          // Attempt to refresh token
          await this.refreshCallback();
          // Retry the request once after refresh
          return this.request<T>(endpoint, options, false);
        } catch (refreshError) {
          // Refresh failed, clear token and throw
          this.tokenProvider?.clearToken();
          throw new AuthServiceError(
            'Session expired - please login again',
            HTTP_STATUS.UNAUTHORIZED
          );
        }
      }

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
 * Uses HTTP-only cookies for secure token storage
 */
class AuthService {
  private client: HttpClient;

  constructor(client: HttpClient) {
    this.client = client;
    // Set up automatic token refresh
    this.client.setRefreshCallback(() => this.refreshToken());
  }

  /**
   * Register a new user
   * Backend sets HTTP-only cookie with token
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

      // Backend sets HTTP-only cookie automatically
      // No need to store token in localStorage
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
   * Backend sets HTTP-only cookie with token
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

      // Backend sets HTTP-only cookie automatically
      // No need to store token in localStorage
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
   * Backend clears HTTP-only cookie
   */
  async logout(): Promise<void> {
    try {
      await this.client.post(API_CONFIG.ENDPOINTS.LOGOUT || '/auth/logout', {});
      // Backend clears cookie via Set-Cookie with Max-Age=0
    } catch (error) {
      // Still clear local state even if request fails
      console.error('Logout request failed:', error);
    }
  }

  /**
   * Get current user info
   */
  async getCurrentUser(): Promise<AuthResponse> {
    return this.client.get<AuthResponse>(API_CONFIG.ENDPOINTS.ME);
  }

  /**
   * Refresh access token using refresh token
   * Backend validates refresh token from HTTP-only cookie
   */
  private async refreshToken(): Promise<void> {
    try {
      await this.client.post(
        API_CONFIG.ENDPOINTS.REFRESH || '/auth/refresh',
        {}
      );
      // Backend sets new access token cookie
    } catch (error) {
      throw new AuthServiceError(
        'Token refresh failed',
        HTTP_STATUS.UNAUTHORIZED
      );
    }
  }
}

/**
 * Default configuration - uses HTTP-only cookies (no TokenProvider needed)
 * Cookies are automatically sent with credentials: 'include'
 * 
 * For custom token management (e.g., in-memory tokens), inject a TokenProvider:
 * 
 * class MemoryTokenProvider implements TokenProvider {
 *   private token: string | null = null;
 *   
 *   async getToken(): Promise<string | null> {
 *     return this.token;
 *   }
 *   
 *   setToken(token: string): void {
 *     this.token = token;
 *   }
 *   
 *   clearToken(): void {
 *     this.token = null;
 *   }
 * }
 * 
 * const tokenProvider = new MemoryTokenProvider();
 * const httpClient = new HttpClient(BASE_URL, TIMEOUT, tokenProvider);
 * const authService = new AuthService(httpClient);
 */
const httpClient = new HttpClient(API_CONFIG.BASE_URL, API_CONFIG.TIMEOUT);
export const authService = new AuthService(httpClient);
export { AuthServiceError, HttpClient };
export type { TokenProvider };