/**
 * Sign Up Page (Server Component)
 * Following Next.js App Router best practices:
 * - Server Component by default (no "use client")
 * - Better performance - less JavaScript sent to client
 * - Better SEO - fully rendered on server
 * - Client components are isolated to interactive parts only
 * 
 * SOLID Principles:
 * - Single Responsibility: Page handles layout and routing, components handle interactivity
 * - Open/Closed: Easy to extend with metadata, analytics, etc.
 * - Dependency Inversion: Depends on component abstractions
 */

import Link from "next/link";
import { XMarkIcon } from "@heroicons/react/24/outline";
import { SignupForm } from "@/components/auth/SignupForm";
import { SocialLoginButtons } from "@/components/auth/SocialLoginButtons";
import type { Metadata } from "next";

// Server-side metadata (SEO optimization)
export const metadata: Metadata = {
  title: "Sign Up | RealTimeFakeCheck",
  description: "Create your account to start verifying content in real-time",
};

/**
 * Signup Page - Server Component
 * Handles page layout, static content, and imports client components for interactivity
 */
export default function SignupPage() {
  return (
    <div className="flex min-h-screen flex-1 items-center justify-center px-4 py-12 bg-gray-50">
      <div className="w-full max-w-md">
        {/* Modal-like Container */}
        <div className="bg-white rounded-2xl shadow-2xl p-8 relative">
          {/* Close Button - Client component would go in a separate file if needed */}
          <Link
            href="/"
            className="absolute right-4 top-4 text-gray-400 hover:text-gray-600 transition-colors"
            aria-label="Close"
          >
            <XMarkIcon className="h-6 w-6" />
          </Link>

          {/* Header - Static Content (Server rendered) */}
          <div className="text-center mb-6">
            <h1 className="text-3xl font-bold text-gray-900">
              Sign Up
            </h1>
            <p className="mt-2 text-sm text-gray-600">
              Already have an account?{' '}
              <Link 
                href="/login" 
                className="font-semibold text-blue-600 hover:text-blue-500 transition-colors"
              >
                Log In
              </Link>
            </p>
          </div>

          {/* Social Login Buttons - Client Component (interactive) */}
          <SocialLoginButtons isLoading={false} />

          {/* Divider - Static Content (Server rendered) */}
          <div className="relative mb-6">
            <div className="absolute inset-0 flex items-center">
              <div className="w-full border-t border-gray-300"></div>
            </div>
            <div className="relative flex justify-center text-sm">
              <span className="px-4 bg-white text-gray-500">or</span>
            </div>
          </div>

          {/* Signup Form - Client Component (interactive) */}
          <SignupForm />

          {/* Privacy Policy Agreement - Static Content (Server rendered) */}
          <p className="mt-6 text-center text-xs text-gray-600">
            By creating this account, you agree to our{' '}
            <Link 
              href="/privacy-policy" 
              className="font-medium text-blue-600 hover:text-blue-500 transition-colors"
            >
              Privacy Policy
            </Link>
            {' '}&{' '}
            <Link 
              href="/cookie-policy" 
              className="font-medium text-blue-600 hover:text-blue-500 transition-colors"
            >
              Cookie Policy
            </Link>
            .
          </p>
        </div>

        {/* Additional Info - Static Content (Server rendered) */}
        <p className="mt-6 text-center text-sm text-gray-600">
          Need help?{' '}
          <Link 
            href="/support" 
            className="font-medium text-blue-600 hover:text-blue-500 transition-colors"
          >
            Contact Support
          </Link>
        </p>
      </div>
    </div>
  );
}
