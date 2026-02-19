/**
 * useLogin Hook
 * Custom hook for login logic following Single Responsibility Principle
 * Encapsulates all login-related state management and business logic
 */

import { useState, useCallback } from 'react';
import { useRouter } from 'next/navigation';
import { authService, AuthServiceError } from '@/services/auth.service';
import { sanitizeInput } from '@/utils/validators';
import type { LoginData } from '@/types/auth.types';

interface UseLoginReturn {
  // Form state
  formData: LoginData;
  rememberMe: boolean;
  
  // UI state
  isLoading: boolean;
  errors: Record<string, string>;
  showPassword: boolean;
  
  // Actions
  handleInputChange: (field: keyof LoginData, value: string) => void;
  togglePasswordVisibility: () => void;
  toggleRememberMe: () => void;
  handleSubmit: (e: React.FormEvent) => Promise<void>;
  clearError: (field: string) => void;
}

export const useLogin = (onSuccess?: () => void): UseLoginReturn => {
  const router = useRouter();
  
  // Form state
  const [formData, setFormData] = useState<LoginData>({
    email: '',
    password: '',
  });
  
  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showPassword, setShowPassword] = useState(false);
  const [rememberMe, setRememberMe] = useState(false);

  /**
   * Handle input change with sanitization
   */
  const handleInputChange = useCallback((field: keyof LoginData, value: string) => {
    const sanitizedValue = field === 'email' 
      ? sanitizeInput(value).toLowerCase() 
      : value;
    
    setFormData((prev) => ({
      ...prev,
      [field]: sanitizedValue,
    }));

    // Clear error for this field when user starts typing
    if (errors[field]) {
      setErrors((prev) => {
        const newErrors = { ...prev };
        delete newErrors[field];
        return newErrors;
      });
    }
  }, [errors]);

  /**
   * Toggle password visibility
   */
  const togglePasswordVisibility = useCallback(() => {
    setShowPassword((prev) => !prev);
  }, []);

  /**
   * Toggle remember me
   */
  const toggleRememberMe = useCallback(() => {
    setRememberMe((prev) => !prev);
  }, []);

  /**
   * Clear specific error
   */
  const clearError = useCallback((field: string) => {
    setErrors((prev) => {
      const newErrors = { ...prev };
      delete newErrors[field];
      return newErrors;
    });
  }, []);

  /**
   * Basic form validation
   */
  const validateForm = (): boolean => {
    const newErrors: Record<string, string> = {};

    if (!formData.email) {
      newErrors.email = 'Email is required';
    } else if (!/\S+@\S+\.\S+/.test(formData.email)) {
      newErrors.email = 'Please enter a valid email address';
    }

    if (!formData.password) {
      newErrors.password = 'Password is required';
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  /**
   * Handle form submission
   */
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Reset errors
    setErrors({});

    // Validate form
    if (!validateForm()) {
      return;
    }

    // Set loading state
    setIsLoading(true);

    try {
      // Call login API
      const response = await authService.login(formData);

      // Success - call onSuccess callback or redirect to dashboard
      if (onSuccess) {
        onSuccess();
      } else {
        router.push('/dashboard');
      }
    } catch (error) {
      // Handle API errors
      if (error instanceof AuthServiceError) {
        if (error.field) {
          // Field-specific error
          setErrors({ [error.field]: error.message });
        } else {
          // General error
          setErrors({ general: error.message });
        }
      } else if (error instanceof Error) {
        setErrors({ general: error.message });
      } else {
        setErrors({ general: 'An unexpected error occurred. Please try again.' });
      }
    } finally {
      setIsLoading(false);
    }
  }, [formData, router, onSuccess]);

  return {
    formData,
    rememberMe,
    isLoading,
    errors,
    showPassword,
    handleInputChange,
    togglePasswordVisibility,
    toggleRememberMe,
    handleSubmit,
    clearError,
  };
};
