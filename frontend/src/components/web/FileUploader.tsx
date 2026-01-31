"use client";

import { useState, useRef, useCallback } from "react";
import { FileRejection, useDropzone } from "react-dropzone";
import * as tus from "tus-js-client";
import clsx from "clsx";
import { toast } from "sonner";

type UploadStatus = "idle" | "uploading" | "success" | "error";

interface UploadCardProps {
  uploadEndpoint: string;        // tus server endpoint
  authToken?: string;            // JWT token (optional)
  maxSizeGB?: number;
  onUploadComplete?: (url: string) => void;
}

export function UploadCard({
  uploadEndpoint,
  authToken,
  maxSizeGB = 10,
  onUploadComplete,
}: UploadCardProps) {
  const [status, setStatus] = useState<UploadStatus>("idle");
  const [progress, setProgress] = useState(0);
  const [fileName, setFileName] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const uploadRef = useRef<tus.Upload | null>(null);

  const startUpload = (file: File) => {
    setStatus("uploading");
    setError(null);
    setFileName(file.name);

    const upload = new tus.Upload(file, {
      endpoint: uploadEndpoint,
      metadata: {
        filename: file.name,
        filetype: file.type,
      },
      headers: authToken
        ? { Authorization: `Bearer ${authToken}` }
        : undefined,
      chunkSize: 5 * 1024 * 1024, // 5MB chunks
      retryDelays: [0, 1000, 3000, 5000, 10000],
      onProgress(bytesUploaded, bytesTotal) {
        const percent = Math.floor((bytesUploaded / bytesTotal) * 100);
        setProgress(percent);
      },
      onSuccess() {
        setProgress(100);
        setStatus("success");
        if (upload.url && onUploadComplete) {
          onUploadComplete(upload.url);
        }
      },
      onError(err) {
        console.error(err);
        setStatus("error");
        setError("Upload failed. Please try again.");
      },
    });

    uploadRef.current = upload;
    upload.start();
  };

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (!acceptedFiles.length) return;
    startUpload(acceptedFiles[0]);
  }, []);

  const onDropRejected = useCallback((fileRejections: FileRejection[]) => {
    if (fileRejections.length > 0) {
      const tooLarge = fileRejections.some(rejection =>
        rejection.file.size > maxSizeGB * 1024 * 1024 * 1024
      );
      if (tooLarge) {
        const errorMsg = `File size exceeds the maximum limit of ${maxSizeGB}GB.`;
        setError(errorMsg);
        toast.error(errorMsg);
        setStatus("error");
        return;
      }
      const toomanyFiles = fileRejections.some(rejection =>
        rejection.errors.some(e => e.code === "too-many-files")
      );
      if (toomanyFiles) {
        const errorMsg = "Only one file can be uploaded at a time.";
        setError(errorMsg);
        toast.error(errorMsg);
        setStatus("error");
        return;
      }
    }
    const rejection = fileRejections[0];
    if (rejection && rejection.errors.length > 0) {
      const errorMessages = rejection.errors.map(e => e.message).join(", ");
      const errorMsg = `File rejected: ${errorMessages}`;
      setError(errorMsg);
      toast.error(errorMsg);
      setStatus("error");
    }
  }, [maxSizeGB]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    onDropRejected,
    accept: { "video/*": [],
      "image/*": [],
      "audio/*": []
    },
    multiple: false,
    maxSize: maxSizeGB * 1024 * 1024 * 1024,
    maxFiles: 1,
  });

  return (
    <div
      {...getRootProps()}
      className={clsx(
        "w-full max-w-md p-6 rounded-xl border-2 border-dashed transition-colors",
        isDragActive
          ? "border-blue-500 bg-blue-50"
          : "border-gray-300 bg-white",
        status === "uploading" && "pointer-events-none"
      )}
    >
      <input {...getInputProps()} />
      <div className="flex flex-col items-center gap-4 text-center">
        {status === "idle" && (
          <>
            <div className="text-lg font-medium">Drop a video here</div>
            <p className="text-sm text-gray-500">
              Or click to browse (max {maxSizeGB}GB)
            </p>
          </>
        )}

        {status === "uploading" && (
          <div className="w-full">
            <div className="mb-1 text-sm font-medium">Uploading {fileName}</div>
            <div className="h-2 w-full rounded bg-gray-200">
              <div
                className="h-2 rounded bg-blue-600 transition-all"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="mt-1 text-xs text-gray-500">{progress}%</div>
          </div>
        )}

        {status === "success" && (
          <div className="text-green-600 font-medium">Upload complete 🎉</div>
        )}

        {status === "error" && (
          <div className="text-red-600 text-sm">{error}</div>
        )}
      </div>
    </div>
  );
}
