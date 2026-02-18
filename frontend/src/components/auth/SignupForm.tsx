"use client";

/**
 * Signup Form Component (Client Component)
 * Following SOLID principles:
 * - Single Responsibility: Only handles form UI and interactivity
 * - Dependency Inversion: Depends on useSignup hook abstraction
 * 
 * This is a client component because it needs:
 * - Form state management
 * - User interactions (button clicks, input changes)
 * - Browser APIs (localStorage)
 */

import Link from "next/link";
import { useSignup } from "@/hooks/useSignup";
import { EyeIcon, EyeSlashIcon } from "@heroicons/react/24/outline";

export function SignupForm() {
    const {
        formData,
        rememberMe,
        isLoading,
        errors,
        showPassword,
        showConfirmPassword,
        passwordStrength,
        handleInputChange,
        togglePasswordVisibility,
        toggleRememberMe,
        handleSubmit,
    } = useSignup();

    // Password strength color mapping
    const strengthColors = {
        0: 'bg-red-500',
        1: 'bg-orange-500',
        2: 'bg-yellow-500',
        3: 'bg-blue-500',
        4: 'bg-green-500',
    };

    const strengthTextColors = {
        0: 'text-red-600',
        1: 'text-orange-600',
        2: 'text-yellow-600',
        3: 'text-blue-600',
        4: 'text-green-600',
    };

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

            {/* Signup Form */}
            <form onSubmit={handleSubmit} className="space-y-4" noValidate>
                {/* Email Field */}
                <div>
                    <label 
                        htmlFor="email" 
                        className="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Email address
                    </label>
                    <input
                        id="email"
                        name="email"
                        type="email"
                        autoComplete="email"
                        required
                        aria-required="true"
                        aria-invalid={!!errors.email}
                        aria-describedby={errors.email ? "email-error" : undefined}
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
                        <p id="email-error" className="mt-1.5 text-sm text-red-600" role="alert">
                            {errors.email}
                        </p>
                    )}
                </div>

                {/* Password Field */}
                <div>
                    <label 
                        htmlFor="password" 
                        className="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Password
                    </label>
                    <div className="relative">
                        <input
                            id="password"
                            name="password"
                            type={showPassword ? "text" : "password"}
                            autoComplete="new-password"
                            required
                            aria-required="true"
                            aria-invalid={!!errors.password}
                            aria-describedby={errors.password ? "password-error" : "password-strength"}
                            disabled={isLoading}
                            className={`block w-full rounded-lg border ${
                                errors.password 
                                    ? 'border-red-500 focus:ring-red-500 focus:border-red-500' 
                                    : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                            } px-4 py-2.5 pr-11 text-gray-900 placeholder:text-gray-400 focus:ring-2 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed transition-colors`}
                            placeholder="Enter password"
                            value={formData.password}
                            onChange={(e) => handleInputChange('password', e.target.value)}
                        />
                        <button
                            type="button"
                            onClick={() => togglePasswordVisibility('password')}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                            aria-label={showPassword ? "Hide password" : "Show password"}
                            disabled={isLoading}
                            tabIndex={-1}
                        >
                            {showPassword ? (
                                <EyeSlashIcon className="h-5 w-5" />
                            ) : (
                                <EyeIcon className="h-5 w-5" />
                            )}
                        </button>
                    </div>
                    
                    {/* Password Strength Indicator */}
                    {formData.password && (
                        <div id="password-strength" className="mt-2">
                            <div className="flex gap-1 mb-1.5">
                                {[0, 1, 2, 3, 4].map((level) => (
                                    <div
                                        key={level}
                                        className={`h-1 flex-1 rounded-full ${
                                            level <= passwordStrength.score
                                                ? strengthColors[passwordStrength.score as keyof typeof strengthColors]
                                                : 'bg-gray-200'
                                        }`}
                                    />
                                ))}
                            </div>
                            <p className={`text-xs font-medium ${strengthTextColors[passwordStrength.score as keyof typeof strengthTextColors]}`}>
                                {passwordStrength.label}
                            </p>
                            {passwordStrength.score < 4 && passwordStrength.feedback.length > 0 && (
                                <ul className="mt-1 text-xs text-gray-600 space-y-0.5">
                                    {passwordStrength.feedback.slice(0, 2).map((feedback: string, idx: number) => (
                                        <li key={idx}>• {feedback}</li>
                                    ))}
                                </ul>
                            )}
                        </div>
                    )}
                    
                    {errors.password && (
                        <p id="password-error" className="mt-1.5 text-sm text-red-600" role="alert">
                            {errors.password}
                        </p>
                    )}
                </div>

                {/* Confirm Password Field */}
                <div>
                    <label 
                        htmlFor="confirmPassword" 
                        className="block text-sm font-medium text-gray-700 mb-1"
                    >
                        Confirm Password
                    </label>
                    <div className="relative">
                        <input
                            id="confirmPassword"
                            name="confirmPassword"
                            type={showConfirmPassword ? "text" : "password"}
                            autoComplete="new-password"
                            required
                            aria-required="true"
                            aria-invalid={!!errors.confirmPassword}
                            aria-describedby={errors.confirmPassword ? "confirm-password-error" : undefined}
                            disabled={isLoading}
                            className={`block w-full rounded-lg border ${
                                errors.confirmPassword 
                                    ? 'border-red-500 focus:ring-red-500 focus:border-red-500' 
                                    : 'border-gray-300 focus:ring-blue-500 focus:border-blue-500'
                            } px-4 py-2.5 pr-11 text-gray-900 placeholder:text-gray-400 focus:ring-2 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed transition-colors`}
                            placeholder="Confirm your password"
                            value={formData.confirmPassword}
                            onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                        />
                        <button
                            type="button"
                            onClick={() => togglePasswordVisibility('confirmPassword')}
                            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors"
                            aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                            disabled={isLoading}
                            tabIndex={-1}
                        >
                            {showConfirmPassword ? (
                                <EyeSlashIcon className="h-5 w-5" />
                            ) : (
                                <EyeIcon className="h-5 w-5" />
                            )}
                        </button>
                    </div>
                    {errors.confirmPassword && (
                        <p id="confirm-password-error" className="mt-1.5 text-sm text-red-600" role="alert">
                            {errors.confirmPassword}
                        </p>
                    )}
                </div>

                {/* Remember Me Checkbox */}
                <div className="flex items-center justify-between">
                    <div className="flex items-center">
                        <input
                            id="remember-me"
                            name="remember-me"
                            type="checkbox"
                            checked={rememberMe}
                            onChange={toggleRememberMe}
                            disabled={isLoading}
                            className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500 focus:ring-2 disabled:opacity-50 disabled:cursor-not-allowed cursor-pointer"
                        />
                        <label 
                            htmlFor="remember-me" 
                            className="ml-2 block text-sm text-gray-700 cursor-pointer select-none"
                        >
                            Remember Me
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
                <div className="pt-2">
                    <button
                        type="submit"
                        disabled={isLoading}
                        className="w-full flex justify-center items-center rounded-lg bg-teal-600 px-4 py-3 text-base font-semibold text-white shadow-sm hover:bg-teal-700 focus:outline-none focus:ring-2 focus:ring-teal-500 focus:ring-offset-2 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200"
                    >
                        {isLoading ? (
                            <span className="flex items-center">
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
                                Creating account...
                            </span>
                        ) : (
                            'Sign Up'
                        )}
                    </button>
                </div>
            </form>
        </>
    );
}
