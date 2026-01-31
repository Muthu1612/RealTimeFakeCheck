"use client"; // <-- marks this as a client component

import { UploadCard } from "@/components/web/FileUploader";

export default function UploadCardWrapper() {
  const TUS_ENDPOINT = "http://localhost:1080/files/";
  const FAKE_JWT = "test-jwt-token";

  const handleUploadComplete = (url: string) => {
    console.log("File uploaded to:", url);
    alert("Upload complete! URL: " + url);
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-gray-50 p-4">
      <UploadCard
        uploadEndpoint={TUS_ENDPOINT}
        authToken={FAKE_JWT}
        maxSizeGB={10}
        onUploadComplete={handleUploadComplete}
      />
    </div>
  );
}
