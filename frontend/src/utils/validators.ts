/**
 * Validation utilities
 * Following Single Responsibility Principle - each function validates one thing
 */

import type { PasswordStrength, ValidationError, ValidationResult, SignupData } from '@/types/auth.types';

/**
 * Validates email format using RFC 5322 standard
 */
export const isValidEmail = (email: string): boolean => {
  const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
  return emailRegex.test(email.trim());
};

/**
 * Validates password strength and returns detailed feedback
 */
export const getPasswordStrength = (password: string): PasswordStrength => {
  let score = 0;
  const feedback: string[] = [];

  if (!password) {
    return {
      score: 0,
      label: 'Very Weak',
      feedback: ['Password is required'],
    };
  }

  // Length check
  if (password.length >= 8) score++;
  else feedback.push('Use at least 8 characters');

  // Uppercase check
  if (/[A-Z]/.test(password)) score++;
  else feedback.push('Add uppercase letters');

  // Lowercase check
  if (/[a-z]/.test(password)) score++;
  else feedback.push('Add lowercase letters');

  // Number check
  if (/\d/.test(password)) score++;
  else feedback.push('Add numbers');

  // Special character check
  if (/[!@#$%^&*()_+\-=\[\]{};':"\\|,.<>\/?]/.test(password)) score++;
  else feedback.push('Add special characters');

  // Adjust score to 0-4 range
  const finalScore = Math.min(Math.floor(score / 1.25), 4) as 0 | 1 | 2 | 3 | 4;

  const labels: Record<number, 'Very Weak' | 'Weak' | 'Fair' | 'Good' | 'Strong'> = {
    0: 'Very Weak',
    1: 'Weak',
    2: 'Fair',
    3: 'Good',
    4: 'Strong',
  };

  return {
    score: finalScore,
    label: labels[finalScore],
    feedback: finalScore < 4 ? feedback : ['Your password is strong!'],
  };
};

/**
 * Validates if password meets minimum requirements
 */
export const isPasswordValid = (password: string): boolean => {
  return password.length >= 8;
};

/**
 * Validates if passwords match
 */
export const doPasswordsMatch = (password: string, confirmPassword: string): boolean => {
  return password === confirmPassword && password.length > 0;
};

/**
 * Comprehensive signup form validation
 * Returns validation result with all errors
 */
export const validateSignupForm = (data: SignupData): ValidationResult => {
  const errors: ValidationError[] = [];

  // Email validation
  if (!data.email) {
    errors.push({
      field: 'email',
      message: 'Email is required',
    });
  } else if (!isValidEmail(data.email)) {
    errors.push({
      field: 'email',
      message: 'Please enter a valid email address',
    });
  }

  // Password validation
  if (!data.password) {
    errors.push({
      field: 'password',
      message: 'Password is required',
    });
  } else if (!isPasswordValid(data.password)) {
    errors.push({
      field: 'password',
      message: 'Password must be at least 8 characters long',
    });
  }

  // Confirm password validation
  if (!data.confirmPassword) {
    errors.push({
      field: 'confirmPassword',
      message: 'Please confirm your password',
    });
  } else if (!doPasswordsMatch(data.password, data.confirmPassword)) {
    errors.push({
      field: 'confirmPassword',
      message: 'Passwords do not match',
    });
  }

  return {
    isValid: errors.length === 0,
    errors,
  };
};

/**
 * Sanitizes user input to prevent XSS
 */
export const sanitizeInput = (input: string): string => {
  return input.trim().replace(/[<>]/g, '');
};
