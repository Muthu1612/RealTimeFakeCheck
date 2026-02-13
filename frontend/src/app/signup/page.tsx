"use client";

/**
 * Sign Up Page Component
 * Following SOLID principles:
 * - Single Responsibility: Only handles presentation/UI
 * - Open/Closed: Can extend without modifying
 * - Dependency Inversion: Depends on useSignup hook abstraction
 */

import Image from "next/image";
import { useSignup } from "@/hooks/useSignup";
import { EyeIcon, EyeSlashIcon } from "@heroicons/react/24/outline";

export default function SignUpPage() {
    const {
        formData,
        isLoading,
        errors,
        showPassword,
        showConfirmPassword,
        passwordStrength,
        handleInputChange,
        togglePasswordVisibility,
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
        <div className="flex min-h-full flex-1 flex-col justify-center px-6 py-12 lg:px-8">
            <div className="sm:mx-auto sm:w-full sm:max-w-md bg-slate-100 rounded-lg p-8 shadow-lg">
                {/* Header */}
                <div>
                    <Image
                        width={100}
                        height={100}
                        alt="RealTimeFakeCheck"
                        src="https://tailwindcss.com/_next/static/media/tailwindcss-mark.d52e9897.svg"
                        className="mx-auto h-10 w-auto"
                    />
                    <h2 className="mt-5 text-center text-2xl font-bold tracking-tight text-gray-900">
                        Create your account
                    </h2>
                    <p className="mt-2 text-center text-sm text-gray-600">
                        Join us to start verifying content in real-time
                    </p>
                </div>

                <div className="mt-8">
                    {/* General Error Message */}
                    {errors.general && (
                        <div 
                            className="mb-4 p-3 rounded-md bg-red-50 border border-red-200"
                            role="alert"
                            aria-live="polite"
                        >
                            <p className="text-sm text-red-800">{errors.general}</p>
                        </div>
                    )}

                    {/* Signup Form */}
                    <form onSubmit={handleSubmit} className="space-y-5" noValidate>
                        {/* Email Field */}
                        <div>
                            <label 
                                htmlFor="email" 
                                className="block text-sm font-medium text-gray-900"
                            >
                                Email address
                            </label>
                            <div className="mt-2">
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
                                    className={`block w-full rounded-md bg-white px-3 py-2 text-base text-gray-900 outline-1 -outline-offset-1 ${
                                        errors.email 
                                            ? 'outline-red-500 focus:outline-red-600' 
                                            : 'outline-gray-300 focus:outline-indigo-600'
                                    } placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 disabled:opacity-50 disabled:cursor-not-allowed`}
                                    placeholder="you@example.com"
                                    value={formData.email}
                                    onChange={(e) => handleInputChange('email', e.target.value)}
                                />
                                {errors.email && (
                                    <p id="email-error" className="mt-1 text-sm text-red-600" role="alert">
                                        {errors.email}
                                    </p>
                                )}
                            </div>
                        </div>

                        {/* Password Field */}
                        <div>
                            <label 
                                htmlFor="password" 
                                className="block text-sm font-medium text-gray-900"
                            >
                                Password
                            </label>
                            <div className="mt-2 relative">
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
                                    className={`block w-full rounded-md bg-white px-3 py-2 pr-10 text-base text-gray-900 outline-1 -outline-offset-1 ${
                                        errors.password 
                                            ? 'outline-red-500 focus:outline-red-600' 
                                            : 'outline-gray-300 focus:outline-indigo-600'
                                    } placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 disabled:opacity-50 disabled:cursor-not-allowed`}
                                    placeholder="••••••••"
                                    value={formData.password}
                                    onChange={(e) => handleInputChange('password', e.target.value)}
                                />
                                <button
                                    type="button"
                                    onClick={() => togglePasswordVisibility('password')}
                                    className="absolute inset-y-0 right-0 flex items-center pr-3"
                                    aria-label={showPassword ? "Hide password" : "Show password"}
                                    disabled={isLoading}
                                >
                                    {showPassword ? (
                                        <EyeSlashIcon className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                                    ) : (
                                        <EyeIcon className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                                    )}
                                </button>
                            </div>
                            
                            {/* Password Strength Indicator */}
                            {formData.password && (
                                <div id="password-strength" className="mt-2">
                                    <div className="flex gap-1 mb-1">
                                        {[0, 1, 2, 3, 4].map((level) => (
                                            <div
                                                key={level}
                                                className={`h-1 flex-1 rounded ${
                                                    level <= passwordStrength.score
                                                        ? strengthColors[passwordStrength.score]
                                                        : 'bg-gray-200'
                                                }`}
                                            />
                                        ))}
                                    </div>
                                    <p className={`text-xs ${strengthTextColors[passwordStrength.score]}`}>
                                        {passwordStrength.label}
                                    </p>
                                    {passwordStrength.feedback.length > 0 && (
                                        <ul className="mt-1 text-xs text-gray-600 list-disc list-inside">
                                            {passwordStrength.feedback.map((feedback, idx) => (
                                                <li key={idx}>{feedback}</li>
                                            ))}
                                        </ul>
                                    )}
                                </div>
                            )}
                            
                            {errors.password && (
                                <p id="password-error" className="mt-1 text-sm text-red-600" role="alert">
                                    {errors.password}
                                </p>
                            )}
                        </div>

                        {/* Confirm Password Field */}
                        <div>
                            <label 
                                htmlFor="confirmPassword" 
                                className="block text-sm font-medium text-gray-900"
                            >
                                Confirm Password
                            </label>
                            <div className="mt-2 relative">
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
                                    className={`block w-full rounded-md bg-white px-3 py-2 pr-10 text-base text-gray-900 outline-1 -outline-offset-1 ${
                                        errors.confirmPassword 
                                            ? 'outline-red-500 focus:outline-red-600' 
                                            : 'outline-gray-300 focus:outline-indigo-600'
                                    } placeholder:text-gray-400 focus:outline-2 focus:-outline-offset-2 disabled:opacity-50 disabled:cursor-not-allowed`}
                                    placeholder="••••••••"
                                    value={formData.confirmPassword}
                                    onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
                                />
                                <button
                                    type="button"
                                    onClick={() => togglePasswordVisibility('confirmPassword')}
                                    className="absolute inset-y-0 right-0 flex items-center pr-3"
                                    aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                                    disabled={isLoading}
                                >
                                    {showConfirmPassword ? (
                                        <EyeSlashIcon className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                                    ) : (
                                        <EyeIcon className="h-5 w-5 text-gray-400 hover:text-gray-600" />
                                    )}
                                </button>
                            </div>
                            {errors.confirmPassword && (
                                <p id="confirm-password-error" className="mt-1 text-sm text-red-600" role="alert">
                                    {errors.confirmPassword}
                                </p>
                            )}
                        </div>

                        {/* Submit Button */}
                        <div>
                            <button
                                type="submit"
                                disabled={isLoading}
                                className="flex w-full justify-center rounded-md bg-indigo-600 px-3 py-2.5 text-base font-semibold text-white shadow-sm hover:bg-indigo-500 focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-indigo-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
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
                                    'Sign up'
                                )}
                            </button>
                        </div>
                    </form>

                    {/* Login Link */}
                    <p className="mt-6 text-center text-sm text-gray-600">
                        Already have an account?{' '}
                        <a 
                            href="/login" 
                            className="font-semibold text-indigo-600 hover:text-indigo-500 transition-colors"
                        >
                            Log in
                        </a>
                    </p>
                </div>
            </div>
        </div>
    );
}