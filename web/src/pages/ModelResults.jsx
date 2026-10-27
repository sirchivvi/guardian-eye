import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
} from 'recharts';
import {
  Layers,
  BarChart3,
  TrendingUp,
  Table,
  Grid,
  PieChart as PieIcon,
  CheckCircle,
  AlertCircle,
  HelpCircle,
  Sparkles,
} from 'lucide-react';
import { Card } from '../components/Card';
import { LoadingSkeleton, ChartSkeleton } from '../components/LoadingSkeleton';
import { fallbackMetrics } from '../data/fallbackMetrics';
import { SproutSvg, OwlSvg, FoxSvg } from '../components/Animals';

export const ModelResults = () => {
  const [metrics, setMetrics] = useState(null);
  const [loading, setLoading] = useState(true);
  const [activeSplit, setActiveSplit] = useState('test'); // 'train' | 'val' | 'test'
  const [curveMode, setCurveMode] = useState('loss'); // 'loss' | 'accuracy'

  useEffect(() => {
    document.title = 'GuardianEye — Model Benchmarks & Results';

    // Fetch /metrics.json with fallback
    fetch('/metrics.json')
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        setMetrics(data);
        setLoading(false);
      })
      .catch((err) => {
        console.warn('Could not load /metrics.json, using built-in fallback data:', err);
        setMetrics(fallbackMetrics);
        setLoading(false);
      });
  }, []);

  if (loading || !metrics) {
    return (
      <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-12 space-y-8">
        <LoadingSkeleton className="h-10 w-64" />
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <ChartSkeleton />
          <ChartSkeleton />
          <ChartSkeleton />
        </div>
      </div>
    );
  }

  const currentSplitData = metrics.splits[activeSplit] || metrics.splits.test;
  const history = metrics.training_history || [];
  const datasets = metrics.dataset_composition || [];

  // Color constants for soft palette
  const COLORS = {
    sage: '#5B8266',
    sageLight: '#76A284',
    terracotta: '#D96B43',
    terracottaLight: '#E07A5F',
    sunny: '#F59E0B',
    amberLight: '#FDE68A',
    dark: '#2D3748',
    cream: '#FFF9F0',
  };

  // Confusion matrix cell color helper (heat intensity)
  const getHeatmapColor = (value, rowTotal) => {
    const ratio = rowTotal > 0 ? value / rowTotal : 0;
    if (ratio > 0.8) return 'bg-[#5B8266] text-white font-black';
    if (ratio > 0.5) return 'bg-[#76A284] text-white font-bold';
    if (ratio > 0.2) return 'bg-[#C4D9CC] text-[#22543D] font-bold';
    if (ratio > 0.05) return 'bg-[#E2ECE5] text-[#2D3748]';
    return 'bg-[#FFF9F0] text-[#718096]';
  };

  return (
    <div className="max-w-6xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-12">
      {/* Page Header */}
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-6 pb-6 border-b border-amber-200/60">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-[#E2ECE5] text-[#385542] text-xs font-bold">
            <BarChart3 className="w-3.5 h-3.5 text-[#5B8266]" />
            <span>Benchmark Performance</span>
          </div>
          <h1 className="text-3xl sm:text-5xl font-extrabold text-[#2D3748] font-display">
            Model Benchmarks & Curves
          </h1>
          <p className="text-sm sm:text-base text-[#4A5568] max-w-2xl">
            Empirical results across training epochs, 4-class camera-trap evaluation, acoustic
            gunshot detection, and temperature calibration.
          </p>
        </div>

        {/* Split Switcher Pills */}
        <div className="flex items-center bg-white p-1.5 rounded-full border border-amber-200/80 shadow-xs">
          <span className="text-xs font-bold text-[#718096] px-3 hidden sm:inline">Split:</span>
          {['train', 'val', 'test'].map((split) => (
            <button
              key={split}
              onClick={() => setActiveSplit(split)}
              className={`px-4 py-1.5 rounded-full text-xs font-extrabold transition-all uppercase tracking-wider ${
                activeSplit === split
                  ? 'bg-[#5B8266] text-white shadow-xs'
                  : 'text-[#4A5568] hover:text-[#2D3748] hover:bg-amber-50'
              }`}
            >
              {split}
            </button>
          ))}
        </div>
      </div>

      {/* Summary KPI Badges */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 sm:gap-6">
        <Card className="p-5 sm:p-6 text-center space-y-1 border border-amber-100">
          <span className="text-xs font-extrabold text-[#718096] uppercase tracking-wider">
            Overall Accuracy
          </span>
          <p className="text-3xl sm:text-4xl font-extrabold text-[#5B8266] font-display">
            {(currentSplitData.overall.accuracy * 100).toFixed(1)}%
          </p>
          <span className="text-xs text-[#718096]">Multiclass Vision</span>
        </Card>

        <Card className="p-5 sm:p-6 text-center space-y-1 border border-amber-100">
          <span className="text-xs font-extrabold text-[#718096] uppercase tracking-wider">
            Macro F1-Score
          </span>
          <p className="text-3xl sm:text-4xl font-extrabold text-[#D96B43] font-display">
            {(currentSplitData.overall.macro_f1 * 100).toFixed(1)}%
          </p>
          <span className="text-xs text-[#718096]">Balanced across classes</span>
        </Card>

        <Card className="p-5 sm:p-6 text-center space-y-1 border border-amber-100">
          <span className="text-xs font-extrabold text-[#718096] uppercase tracking-wider">
            Gunshot Audio F1
          </span>
          <p className="text-3xl sm:text-4xl font-extrabold text-[#F59E0B] font-display">
            {(currentSplitData.audio_metrics.overall.f1 * 100).toFixed(1)}%
          </p>
          <span className="text-xs text-[#718096]">Acoustic gunshot branch</span>
        </Card>

        <Card className="p-5 sm:p-6 text-center space-y-1 border border-amber-100">
          <span className="text-xs font-extrabold text-[#718096] uppercase tracking-wider">
            Evaluated Samples
          </span>
          <p className="text-3xl sm:text-4xl font-extrabold text-[#2D3748] font-display">
            {currentSplitData.overall.samples?.toLocaleString() || '1,852'}
          </p>
          <span className="text-xs text-[#718096] uppercase">In {activeSplit} split</span>
        </Card>
      </div>

      {/* Training & Validation Curves Section */}
      <Card className="space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2">
          <div>
            <div className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5 text-[#5B8266]" />
              <h3 className="text-xl font-extrabold text-[#2D3748] font-display">
                Training & Validation Convergence Curves
              </h3>
            </div>
            <p className="text-xs sm:text-sm text-[#718096]">
              30 epochs of training with Adam optimizer (lr=1e-4) and class-balanced cross-entropy
            </p>
          </div>

          {/* Toggle Loss vs Accuracy */}
          <div className="flex items-center bg-amber-50 p-1 rounded-full border border-amber-200">
            <button
              onClick={() => setCurveMode('loss')}
              className={`px-3 py-1 rounded-full text-xs font-bold transition-all ${
                curveMode === 'loss' ? 'bg-[#D96B43] text-white shadow-xs' : 'text-[#78350F]'
              }`}
            >
              Cross-Entropy Loss
            </button>
            <button
              onClick={() => setCurveMode('accuracy')}
              className={`px-3 py-1 rounded-full text-xs font-bold transition-all ${
                curveMode === 'accuracy' ? 'bg-[#5B8266] text-white shadow-xs' : 'text-[#78350F]'
              }`}
            >
              Validation Accuracy
            </button>
          </div>
        </div>

        {/* Recharts LineChart */}
        <div className="h-72 sm:h-80 w-full pt-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#F4EDE4" vertical={false} />
              <XAxis
                dataKey="epoch"
                tick={{ fill: '#718096', fontSize: 12 }}
                tickLine={false}
                label={{ value: 'Epoch', position: 'insideBottom', offset: -4, fill: '#718096', fontSize: 12 }}
              />
              <YAxis
                tick={{ fill: '#718096', fontSize: 12 }}
                tickLine={false}
                domain={curveMode === 'loss' ? [0, 1.5] : [0.4, 1.0]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#FFF9F0',
                  borderRadius: '16px',
                  border: '1px solid #E2ECE5',
                  boxShadow: '0 8px 25px rgba(0,0,0,0.06)',
                  fontFamily: 'Nunito, sans-serif',
                  fontSize: '12px',
                }}
              />
              <Legend wrapperStyle={{ paddingTop: '10px', fontSize: '12px' }} />

              {curveMode === 'loss' ? (
                <>
                  <Line
                    type="monotone"
                    dataKey="image_train_loss"
                    name="Image Train Loss"
                    stroke={COLORS.sage}
                    strokeWidth={2.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="image_val_loss"
                    name="Image Val Loss"
                    stroke={COLORS.sageLight}
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="audio_train_loss"
                    name="Audio Train Loss"
                    stroke={COLORS.terracotta}
                    strokeWidth={2.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="audio_val_loss"
                    name="Audio Val Loss"
                    stroke={COLORS.terracottaLight}
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                </>
              ) : (
                <>
                  <Line
                    type="monotone"
                    dataKey="image_train_acc"
                    name="Image Train Acc"
                    stroke={COLORS.sage}
                    strokeWidth={2.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="image_val_acc"
                    name="Image Val Acc"
                    stroke={COLORS.sageLight}
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="audio_train_acc"
                    name="Audio Train Acc"
                    stroke={COLORS.sunny}
                    strokeWidth={2.5}
                    dot={false}
                  />
                  <Line
                    type="monotone"
                    dataKey="audio_val_acc"
                    name="Audio Val Acc"
                    stroke={COLORS.terracotta}
                    strokeWidth={2}
                    strokeDasharray="4 4"
                    dot={false}
                  />
                </>
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </Card>

      {/* Grid: Per-Class Table + Confusion Matrix */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Per-Class Metrics Table (7 cols) */}
        <Card className="lg:col-span-7 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Table className="w-5 h-5 text-[#5B8266]" />
              <h3 className="text-xl font-extrabold text-[#2D3748] font-display">
                Per-Class Performance
              </h3>
            </div>
            <span className="text-xs uppercase font-extrabold px-3 py-1 rounded-full bg-[#E2ECE5] text-[#385542]">
              {activeSplit} split
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left text-sm">
              <thead>
                <tr className="border-b border-amber-100 text-xs text-[#718096] uppercase font-bold">
                  <th className="py-3 px-2">Class</th>
                  <th className="py-3 px-2 text-right">Precision</th>
                  <th className="py-3 px-2 text-right">Recall</th>
                  <th className="py-3 px-2 text-right">F1-Score</th>
                  <th className="py-3 px-2 text-right">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-amber-100/60 font-semibold text-[#2D3748]">
                {currentSplitData.per_class.map((c) => (
                  <tr key={c.name} className="hover:bg-amber-50/40 transition-colors">
                    <td className="py-3 px-2 capitalize flex items-center gap-2">
                      <span className="text-base">
                        {c.name === 'human'
                          ? '🚶'
                          : c.name === 'vehicle'
                          ? '🚙'
                          : c.name === 'animal'
                          ? '🦌'
                          : '🍃'}
                      </span>
                      <span>{c.name}</span>
                    </td>
                    <td className="py-3 px-2 text-right font-mono">
                      {(c.precision * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-2 text-right font-mono">
                      {(c.recall * 100).toFixed(1)}%
                    </td>
                    <td className="py-3 px-2 text-right">
                      <div className="inline-flex items-center gap-2">
                        <div className="w-14 h-2 bg-amber-100 rounded-full overflow-hidden hidden sm:inline-block">
                          <div
                            className="h-full bg-[#5B8266] rounded-full"
                            style={{ width: `${c.f1 * 100}%` }}
                          />
                        </div>
                        <span className="font-mono font-bold text-[#5B8266]">
                          {(c.f1 * 100).toFixed(1)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-3 px-2 text-right text-xs text-[#718096]">
                      {c.support.toLocaleString()}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Audio Branch Sub-Table */}
          <div className="pt-4 border-t border-amber-100/70">
            <h4 className="text-xs font-bold uppercase tracking-wider text-[#D96B43] mb-2">
              Acoustic Gunshot Detection Classes
            </h4>
            <div className="grid grid-cols-2 gap-3 text-xs">
              {currentSplitData.audio_metrics.per_class.map((ac) => (
                <div
                  key={ac.name}
                  className="p-3 rounded-2xl bg-[#FFF9F0] border border-amber-100 flex items-center justify-between"
                >
                  <span className="font-extrabold capitalize text-[#2D3748] flex items-center gap-1.5">
                    <span>{ac.name === 'gunshot' ? '💥' : '🌿'}</span>
                    <span>{ac.name}</span>
                  </span>
                  <span className="font-mono font-bold text-[#D96B43]">
                    F1: {(ac.f1 * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Confusion Matrix Heatmap (5 cols) */}
        <Card className="lg:col-span-5 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Grid className="w-5 h-5 text-[#D96B43]" />
              <h3 className="text-xl font-extrabold text-[#2D3748] font-display">
                Confusion Matrix
              </h3>
            </div>
            <span className="text-xs text-[#718096]">True vs. Predicted</span>
          </div>

          <p className="text-xs text-[#718096]">
            Rows represent ground-truth classes; columns represent predicted classes.
          </p>

          {/* Heatmap Grid */}
          <div className="pt-2">
            <div className="grid grid-cols-5 gap-1.5 text-center text-xs font-bold">
              {/* Header corner */}
              <div className="text-[10px] text-[#718096] uppercase flex items-center justify-center">
                True \ Pred
              </div>
              {currentSplitData.confusion_matrix.classes.map((cls) => (
                <div key={cls} className="py-1 text-[11px] text-[#718096] font-extrabold truncate">
                  {cls}
                </div>
              ))}

              {/* Rows */}
              {currentSplitData.confusion_matrix.matrix.map((row, rIdx) => {
                const rowLabel = currentSplitData.confusion_matrix.classes[rIdx];
                const rowTotal = row.reduce((a, b) => a + b, 0);

                return (
                  <React.Fragment key={rowLabel}>
                    <div className="py-2.5 text-[11px] text-[#718096] font-extrabold text-right pr-2 truncate">
                      {rowLabel}
                    </div>
                    {row.map((val, cIdx) => (
                      <div
                        key={`${rIdx}-${cIdx}`}
                        title={`True: ${rowLabel}, Predicted: ${currentSplitData.confusion_matrix.classes[cIdx]} (${val} samples)`}
                        className={`py-2.5 rounded-xl transition-transform hover:scale-105 flex flex-col items-center justify-center cursor-default ${getHeatmapColor(
                          val,
                          rowTotal
                        )}`}
                      >
                        <span className="text-xs leading-none">{val}</span>
                        <span className="text-[9px] opacity-75 leading-tight">
                          {((val / rowTotal) * 100).toFixed(0)}%
                        </span>
                      </div>
                    ))}
                  </React.Fragment>
                );
              })}
            </div>
          </div>

          {/* Audio 2x2 Matrix Mini */}
          <div className="pt-4 border-t border-amber-100/70">
            <div className="flex items-center justify-between mb-2">
              <span className="text-xs font-bold uppercase text-[#718096]">
                Audio Matrix (Gunshot vs Ambient)
              </span>
              <span className="text-xs text-[#5B8266] font-bold">
                Accuracy: {(currentSplitData.audio_metrics.overall.accuracy * 100).toFixed(1)}%
              </span>
            </div>
            <div className="grid grid-cols-3 gap-1.5 text-center text-xs">
              <div />
              <div className="text-[10px] text-[#718096] font-bold">Gunshot</div>
              <div className="text-[10px] text-[#718096] font-bold">Ambient</div>

              <div className="text-[10px] text-[#718096] font-bold text-right pr-1">Gunshot</div>
              <div className="py-2 rounded-lg bg-[#D96B43] text-white font-black">
                {currentSplitData.audio_metrics.confusion_matrix.matrix[0][0]}
              </div>
              <div className="py-2 rounded-lg bg-[#FFF9F0] text-[#718096]">
                {currentSplitData.audio_metrics.confusion_matrix.matrix[0][1]}
              </div>

              <div className="text-[10px] text-[#718096] font-bold text-right pr-1">Ambient</div>
              <div className="py-2 rounded-lg bg-[#FFF9F0] text-[#718096]">
                {currentSplitData.audio_metrics.confusion_matrix.matrix[1][0]}
              </div>
              <div className="py-2 rounded-lg bg-[#5B8266] text-white font-black">
                {currentSplitData.audio_metrics.confusion_matrix.matrix[1][1]}
              </div>
            </div>
          </div>
        </Card>
      </div>

      {/* Dataset Composition & Temperature Calibration Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
        {/* Dataset Composition (7 cols) */}
        <Card className="lg:col-span-7 space-y-6">
          <div className="flex items-center gap-2">
            <PieIcon className="w-5 h-5 text-[#F59E0B]" />
            <div>
              <h3 className="text-xl font-extrabold text-[#2D3748] font-display">
                Dataset Composition & Modalities
              </h3>
              <p className="text-xs text-[#718096]">
                Trained across four diverse field sources (Caltech, BIRDSAI, Belize, Vietnam)
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
            {/* Pie Chart */}
            <div className="h-56 w-full">
              <ResponsiveContainer width="100%" height="100%">
                <PieChart>
                  <Pie
                    data={datasets}
                    cx="50%"
                    cy="50%"
                    innerRadius={50}
                    outerRadius={80}
                    paddingAngle={4}
                    dataKey="samples"
                  >
                    {datasets.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </Pie>
                  <Tooltip
                    contentStyle={{
                      backgroundColor: '#FFF9F0',
                      borderRadius: '12px',
                      border: '1px solid #E2ECE5',
                      fontSize: '12px',
                    }}
                  />
                </PieChart>
              </ResponsiveContainer>
            </div>

            {/* Legend list */}
            <div className="space-y-3">
              {datasets.map((d) => (
                <div key={d.name} className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span
                      className="w-3 h-3 rounded-full flex-shrink-0"
                      style={{ backgroundColor: d.color }}
                    />
                    <span className="font-extrabold text-[#2D3748]">{d.name}</span>
                  </div>
                  <span className="font-mono text-[#718096] font-bold">
                    {d.samples.toLocaleString()}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </Card>

        {/* Temperature Scaling Calibration Card (5 cols) */}
        <Card className="lg:col-span-5 space-y-4 bg-gradient-to-br from-white to-[#F2F7F4] border border-[#C4D9CC]">
          <div className="flex items-center gap-2">
            <Sparkles className="w-5 h-5 text-[#5B8266]" />
            <h3 className="text-xl font-extrabold text-[#2D3748] font-display">
              Confidence Calibration
            </h3>
          </div>
          <p className="text-xs text-[#4A5568] leading-relaxed">
            Neural networks are notorious for overconfidence. Stage 2A fits strictly positive
            temperatures (<span className="font-mono font-bold">T &gt; 0</span>) on validation data
            to ensure predicted probabilities accurately mirror real-world uncertainty.
          </p>

          <div className="space-y-3 pt-2">
            <div className="p-3.5 rounded-2xl bg-white border border-[#C4D9CC] flex items-center justify-between">
              <div>
                <p className="text-xs text-[#718096] font-bold">Image Temperature (T_img)</p>
                <p className="text-lg font-mono font-extrabold text-[#5B8266]">
                  {metrics.calibration?.image_temperature || '1.28'}
                </p>
              </div>
              <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-[#E2ECE5] text-[#385542]">
                Validated
              </span>
            </div>

            <div className="p-3.5 rounded-2xl bg-white border border-[#C4D9CC] flex items-center justify-between">
              <div>
                <p className="text-xs text-[#718096] font-bold">Audio Temperature (T_aud)</p>
                <p className="text-lg font-mono font-extrabold text-[#D96B43]">
                  {metrics.calibration?.audio_temperature || '1.15'}
                </p>
              </div>
              <span className="text-[11px] font-bold px-2 py-0.5 rounded-full bg-[#F9E5DE] text-[#9C3F1D]">
                Validated
              </span>
            </div>
          </div>

          <div className="pt-2 text-xs text-[#718096] flex items-center justify-between">
            <span>Expected Calibration Error:</span>
            <span className="font-bold text-[#5B8266]">
              {metrics.calibration?.ece_before || '8.4%'} → {metrics.calibration?.ece_after || '2.1%'}
            </span>
          </div>
        </Card>
      </div>
    </div>
  );
};
