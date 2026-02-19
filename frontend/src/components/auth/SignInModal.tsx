"use client";

/**
 * Sign In Modal Component
 * A popup modal for user authentication
 * Can be triggered from anywhere in the application
 */

import { XMarkIcon } from "@heroicons/react/24/outline";
import { LoginForm } from "./LoginForm";
import { SocialLoginButtons } from "./SocialLoginButtons";
import { useEffect } from "react";

interface SignInModalProps {
  isOpen: boolean;
  onClose: () => void;
  onSwitchToSignUp?: () => void;
}

export function SignInModal({ isOpen, onClose, onSwitchToSignUp }: SignInModalProps) {
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
      <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
        <div 
          className="bg-white rounded-2xl shadow-2xl w-full max-w-md relative animate-in fade-in zoom-in duration-200"
          role="dialog"
          aria-modal="true"
          aria-labelledby="signin-modal-title"
          onClick={(e) => e.stopPropagation()}
        >
          {/* Close Button */}
          <button
            onClick={onClose}
            className="absolute right-4 top-4 text-gray-400 hover:text-gray-600 transition-colors z-10"
            aria-label="Close sign in modal"
          >
            <XMarkIcon className="h-6 w-6" />
          </button>

          {/* Modal Content */}
          <div className="p-8">
            {/* Header */}
            <div className="text-center mb-6">
              <h2 
                id="signin-modal-title"
                className="text-3xl font-bold text-gray-900"
              >
                Sign In
              </h2>
              <p className="mt-2 text-sm text-gray-600">
                Welcome back! Please sign in to continue
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

            {/* Login Form */}
            <LoginForm onSuccess={onClose} />

            {/* Sign Up Link */}
            <p className="mt-6 text-center text-sm text-gray-600">
              Don't have an account?{' '}
              {onSwitchToSignUp ? (
                <button
                  onClick={onSwitchToSignUp}
                  className="font-semibold text-blue-600 hover:text-blue-500 transition-colors"
                >
                  Sign Up
                </button>
              ) : (
                <a 
                  href="/signup" 
                  className="font-semibold text-blue-600 hover:text-blue-500 transition-colors"
                  onClick={onClose}
                >
                  Sign Up
                </a>
              )}
            </p>
          </div>
        </div>
      </div>
    </>
  );
}
