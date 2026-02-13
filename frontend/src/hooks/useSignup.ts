/**
 * useSignup Hook
 * Custom hook for signup logic following Single Responsibility Principle
 * Encapsulates all signup-related state management and business logic
 */

import { useState, useCallback, useMemo } from 'react';
import { useRouter } from 'next/navigation';
import { authService, AuthServiceError } from '@/services/auth.service';
import { validateSignupForm, getPasswordStrength, sanitizeInput } from '@/utils/validators';
import type { SignupData, ValidationError, PasswordStrength } from '@/types/auth.types';

interface UseSignupReturn {
  // Form state
  formData: SignupData;
  
  // UI state
  isLoading: boolean;
  errors: Record<string, string>;
  showPassword: boolean;
  showConfirmPassword: boolean;
  
  // Password strength
  passwordStrength: PasswordStrength;
  
  // Actions
  handleInputChange: (field: keyof SignupData, value: string) => void;
  togglePasswordVisibility: (field: 'password' | 'confirmPassword') => void;
  handleSubmit: (e: React.FormEvent) => Promise<void>;
  clearError: (field: string) => void;
}

export const useSignup = (): UseSignupReturn => {
  const router = useRouter();
  
  // Form state
  const [formData, setFormData] = useState<SignupData>({
    email: '',
    password: '',
    confirmPassword: '',
  });
  
  // UI state
  const [isLoading, setIsLoading] = useState(false);
  const [errors, setErrors] = useState<Record<string, string>>({});
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Memoized password strength calculation
  const passwordStrength = useMemo(
    () => getPasswordStrength(formData.password),
    [formData.password]
  );

  /**
   * Handle input change with sanitization
   */
  const handleInputChange = useCallback((field: keyof SignupData, value: string) => {
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
  const togglePasswordVisibility = useCallback((field: 'password' | 'confirmPassword') => {
    if (field === 'password') {
      setShowPassword((prev) => !prev);
    } else {
      setShowConfirmPassword((prev) => !prev);
    }
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
   * Convert validation errors to error object
   */
  const validationErrorsToObject = (validationErrors: ValidationError[]): Record<string, string> => {
    return validationErrors.reduce((acc, error) => {
      acc[error.field] = error.message;
      return acc;
    }, {} as Record<string, string>);
  };

  /**
   * Handle form submission
   */
  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Reset errors
    setErrors({});

    // Validate form
    const validation = validateSignupForm(formData);
    if (!validation.isValid) {
      setErrors(validationErrorsToObject(validation.errors));
      return;
    }

    // Set loading state
    setIsLoading(true);

    try {
      // Call signup API
      const response = await authService.signup({
        email: formData.email,
        password: formData.password,
      });

      // Success - redirect to dashboard
      router.push('/dashboard');
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
  }, [formData, router]);

  return {
    formData,
    isLoading,
    errors,
    showPassword,
    showConfirmPassword,
    passwordStrength,
    handleInputChange,
    togglePasswordVisibility,
    handleSubmit,
    clearError,
  };
};
