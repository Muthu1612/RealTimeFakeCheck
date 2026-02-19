"use client";

/**
 * Login Form Component (Client Component)
 * Following SOLID principles:
 * - Single Responsibility: Only handles login form UI and interactivity
 * - Dependency Inversion: Depends on useLogin hook abstraction
 */

import Link from "next/link";
import { useLogin } from "@/hooks/useLogin";
import { EyeIcon, EyeSlashIcon } from "@heroicons/react/24/outline";

interface LoginFormProps {
  onSuccess?: () => void;
}

export function LoginForm({ onSuccess }: LoginFormProps) {
  const {
    formData,
    rememberMe,
    isLoading,
    errors,
    showPassword,
    handleInputChange,
    togglePasswordVisibility,
    toggleRememberMe,
    handleSubmit,
  } = useLogin(onSuccess);

  return (
    <>
      {/* General Error Message */}
      {errors.general && (
        <div 
          className="mb-4 p-3 rounded-lg bg-red-50 border border-red-200"
          role="alert"
          aria-live="polite"
        >
          <p className="text-sm text-red-800">{errors.general}</p>
        </div>
      )}

      {/* Login Form */}
      <form onSubmit={handleSubmit} className="space-y-4" noValidate>
        {/* Email Field */}
        <div>
          <label 
            htmlFor="login-email" 
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Email address
          </label>
          <input
            id="login-email"
            name="email"
            type="email"
            autoComplete="email"
            required
            aria-required="true"
            aria-invalid={!!errors.email}
            aria-describedby={errors.email ? "login-email-error" : undefined}
            disabled={isLoading}
            className={`block w-full rounded-lg border ${
              errors.email 
                ? 'border-red-500 focus:ring-red-500 focus:border-red-500' 
                : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
            } px-4 py-2.5 text-gray-900 placeholder:text-gray-400 focus:ring-2 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed transition-colors`}
            placeholder="Enter your email"
            value={formData.email}
            onChange={(e) => handleInputChange('email', e.target.value)}
          />
          {errors.email && (
            <p id="login-email-error" className="mt-1.5 text-sm text-red-600" role="alert">
              {errors.email}
            </p>
          )}
        </div>

        {/* Password Field */}
        <div>
          <label 
            htmlFor="login-password" 
            className="block text-sm font-medium text-gray-700 mb-1"
          >
            Password
          </label>
          <div className="relative">
            <input
              id="login-password"
              name="password"
              type={showPassword ? 'text' : 'password'}
              autoComplete="current-password"
              required
              aria-required="true"
              aria-invalid={!!errors.password}
              aria-describedby={errors.password ? "login-password-error" : undefined}
              disabled={isLoading}
              className={`block w-full rounded-lg border ${
                errors.password 
                  ? 'border-red-500 focus:ring-red-500 focus:border-red-500' 
                  : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
              } px-4 py-2.5 pr-12 text-gray-900 placeholder:text-gray-400 focus:ring-2 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed transition-colors`}
              placeholder="Enter your password"
              value={formData.password}
              onChange={(e) => handleInputChange('password', e.target.value)}
            />
            <button
              type="button"
              onClick={togglePasswordVisibility}
              disabled={isLoading}
              className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 focus:outline-none focus:text-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
              aria-label={showPassword ? 'Hide password' : 'Show password'}
            >
              {showPassword ? (
                <EyeSlashIcon className="h-5 w-5" />
              ) : (
                <EyeIcon className="h-5 w-5" />
              )}
            </button>
          </div>
          {errors.password && (
            <p id="login-password-error" className="mt-1.5 text-sm text-red-600" role="alert">
              {errors.password}
            </p>
          )}
        </div>

        {/* Remember Me and Forgot Password */}
        <div className="flex items-center justify-between">
          <div className="flex items-center">
            <input
              id="remember-me"
              name="remember-me"
              type="checkbox"
              checked={rememberMe}
              onChange={toggleRememberMe}
              disabled={isLoading}
              className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed"
            />
            <label 
              htmlFor="remember-me" 
              className="ml-2 block text-sm text-gray-700"
            >
              Remember me
            </label>
          </div>

          <Link 
            href="/forgot-password" 
            className="text-sm font-medium text-blue-600 hover:text-blue-500 transition-colors"
          >
            Forgot password?
          </Link>
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={isLoading}
          className="w-full flex justify-center items-center rounded-lg bg-blue-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {isLoading ? (
            <>
              <svg 
                className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" 
                xmlns="http://www.w3.org/2000/svg" 
                fill="none" 
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <circle 
                  className="opacity-25" 
                  cx="12" 
                  cy="12" 
                  r="10" 
                  stroke="currentColor" 
                  strokeWidth="4"
                />
                <path 
                  className="opacity-75" 
                  fill="currentColor" 
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                />
              </svg>
              Signing in...
            </>
          ) : (
            'Sign In'
          )}
        </button>
      </form>
    </>
  );
}
