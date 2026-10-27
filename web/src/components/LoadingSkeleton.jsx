import React from 'react';

export const LoadingSkeleton = ({ className = 'h-6 w-full', rounded = 'rounded-xl' }) => (
  <div
    className={`animate-pulse bg-gradient-to-r from-amber-100/60 via-amber-50 to-amber-100/60 ${rounded} ${className}`}
  />
);

export const CardSkeleton = () => (
  <div className="bg-white rounded-3xl p-6 border border-amber-100 shadow-[0_8px_30px_rgb(0,0,0,0.03)] space-y-4">
    <div className="flex items-center space-x-3">
      <LoadingSkeleton className="h-10 w-10" rounded="rounded-2xl" />
      <div className="space-y-2 flex-1">
        <LoadingSkeleton className="h-4 w-1/3" />
        <LoadingSkeleton className="h-3 w-1/2" />
      </div>
    </div>
    <LoadingSkeleton className="h-32 w-full" rounded="rounded-2xl" />
    <div className="flex gap-2 pt-2">
      <LoadingSkeleton className="h-8 w-24" rounded="rounded-full" />
      <LoadingSkeleton className="h-8 w-24" rounded="rounded-full" />
    </div>
  </div>
);

export const ChartSkeleton = () => (
  <div className="bg-white rounded-3xl p-6 border border-amber-100 shadow-[0_8px_30px_rgb(0,0,0,0.03)] space-y-4">
    <div className="flex justify-between items-center">
      <LoadingSkeleton className="h-6 w-48" />
      <LoadingSkeleton className="h-8 w-32" rounded="rounded-full" />
    </div>
    <LoadingSkeleton className="h-64 w-full" rounded="rounded-2xl" />
  </div>
);
