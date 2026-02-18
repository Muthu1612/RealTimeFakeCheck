/**
 * Authentication and User-related TypeScript types
 * Following Single Responsibility Principle - dedicated type definitions
 */

export interface User {
  id: string;
  email: string;
  name?: string;
  createdAt?: string;
}

export interface SignupData {
  email: string;
  password: string;
  confirmPassword: string;
}

export interface SignupInput {
  email: string;
  password: string;
}

export interface LoginData {
  email: string;
  password: string;
}

export interface AuthResponse {
  user: User;
  token: string;
  message?: string;
}

export interface ApiError {
  message: string;
  field?: string;
  code?: string;
}

export interface ValidationError {
  field: keyof SignupData | keyof LoginData;
  message: string;
}

export interface ValidationResult {
  isValid: boolean;
  errors: ValidationError[];
}

export interface PasswordStrength {
  score: 0 | 1 | 2 | 3 | 4;
  label: 'Very Weak' | 'Weak' | 'Fair' | 'Good' | 'Strong';
  feedback: string[];
}

export interface SignupResult {
  userId: string;
}

export interface SignupError {
  field?: "email" | "password";
  message: string;
}