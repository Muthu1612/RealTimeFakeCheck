"use client";

/**
 * Sign Up Modal Component
 * A popup modal for user registration
 * Can be triggered from anywhere in the application
 */

import { XMarkIcon } from "@heroicons/react/24/outline";
import { SignupForm } from "./SignupForm";
import { SocialLoginButtons } from "./SocialLoginButtons";
import { useEffect } from "react";
import Link from "next/link";

interface SignUpModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSwitchToSignIn?: () => void;
}

export function SignUpModal({ isOpen, onClose, onSwitchToSignIn }: SignUpModalProps) {
  // Handle escape key to close modal
  useEffect(() => {
    const handleEscape = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) {
        onClose();
      }
    };

    if (isOpen) {
      document.addEventListener('keydown', handleEscape);
      // Prevent body scroll when modal is open
      document.body.style.overflow = 'hidden';
    }

    return () => {
      document.removeEventListener('keydown', handleEscape);
      document.body.style.overflow = 'unset';
    };
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div 
        className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 transition-opacity"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Modal Container */}
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4 overflow-y-auto">
        <div 
          className="bg-white rounded-2xl shadow-2xl w-full max-w-md relative animate-in fade-in zoom-in duration-200 my-8"
          role="dialog"
          aria-modal="true"
          aria-labelledby="signup-modal-title"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Close Button */}
          <button
            onClick={onClose}
            className="absolute right-4 top-4 text-gray-400 hover:text-gray-600 transition-colors z-10"
            aria-label="Close sign up modal"
          >
            <XMarkIcon className="h-6 w-6" />
          </button>

          {/* Modal Content */}
          <div className="p-8">
            {/* Header */}
            <div className="text-center mb-6">
              <h2 
                id="signup-modal-title"
                className="text-3xl font-bold text-gray-900"
              >
                Sign Up
              </h2>
              <p className="mt-2 text-sm text-gray-600">
                Create your account to start verifying content
              </p>
            </div>

            {/* Social Login Buttons */}
            <SocialLoginButtons isLoading={false} />

            {/* Divider */}
            <div className="relative my-6">
              <div className="absolute inset-0 flex items-center">
                <div className="w-full border-t border-gray-300"></div>
              </div>
              <div className="relative flex justify-center text-sm">
                <span className="px-4 bg-white text-gray-500">or</span>
              </div>
            </div>

            {/* Signup Form */}
            <SignupForm />

            {/* Privacy Policy Agreement */}
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

            {/* Sign In Link */}
            <p className="mt-4 text-center text-sm text-gray-600">
              Already have an account?{' '}
              {onSwitchToSignIn ? (
                <button 
                  onClick={onSwitchToSignIn}
                  className="font-semibold text-blue-600 hover:text-blue-500 transition-colors"
                >
                  Sign In
                </button>
              ) : (
                <a 
                  href="/login" 
                  className="font-semibold text-blue-600 hover:text-blue-500 transition-colors"
                  onClick={onClose}
                >
                  Sign In
                </a>
              )}
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
