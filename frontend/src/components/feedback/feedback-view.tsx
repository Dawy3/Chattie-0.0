'use client'

import { useQuery } from '@tanstack/react-query'
import { TrendingUp, RefreshCw, Loader2, Database, Server, Users } from 'lucide-react'
import { getHealth, getHealthStats, type HealthStatus, type StatsResponse } from '@/lib/api'
import { clsx } from 'clsx'

export function FeedbackView() {
  const { data: health, isLoading: healthLoading, refetch: refetchHealth } = useQuery({
    queryKey: ['health'],
    queryFn: getHealth,
    refetchInterval: 30000,
  })

  const { data: stats, isLoading: statsLoading, refetch: refetchStats } = useQuery({
    queryKey: ['health-stats'],
    queryFn: getHealthStats,
    refetchInterval: 30000,
  })

  const handleRefresh = () => {
    refetchHealth()
    refetchStats()
  }

  return (
    <div className="h-full overflow-y-auto p-6">
      <div className="max-w-4xl mx-auto space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">System Health</h1>
            <p className="text-sm text-muted-foreground">
              Monitor system status and statistics
            </p>
          </div>
          <button
            onClick={handleRefresh}
            className="p-2 rounded-lg hover:bg-muted transition-colors"
            title="Refresh"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>

        {/* Health Status */}
        <div className="rounded-xl border border-border p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            API Status
          </h2>
          {healthLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : health ? (
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              <HealthCard label="Status" value={health.status} />
              <HealthCard label="Version" value={health.version} />
              <HealthCard label="Environment" value={health.environment} />
            </div>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-4">
              Unable to fetch health status
            </p>
          )}
        </div>

        {/* System Stats */}
        <div className="rounded-xl border border-border p-6">
          <h2 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Server className="w-5 h-5" />
            System Statistics
          </h2>
          {statsLoading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="w-6 h-6 animate-spin text-muted-foreground" />
            </div>
          ) : stats ? (
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <MetricCard
                label="Qdrant Points"
                value={stats.qdrant_points}
                icon={Database}
              />
              <MetricCard
                label="Qdrant Status"
                value={stats.qdrant_status}
                icon={Database}
              />
              <MetricCard
                label="BM25 Index Size"
                value={stats.bm25_index_size}
                icon={Server}
              />
              <MetricCard
                label="Active Sessions"
                value={stats.active_sessions}
                icon={Users}
              />
            </div>
          ) : (
            <p className="text-sm text-muted-foreground text-center py-4">
              Unable to fetch system statistics
            </p>
          )}
        </div>

        {/* Tips */}
        <div className="rounded-xl border border-border p-6 bg-muted/30">
          <h2 className="text-lg font-semibold mb-3">Tips for Improving Quality</h2>
          <ul className="space-y-2 text-sm text-muted-foreground">
            <li className="flex items-start gap-2">
              <span className="text-foreground">1.</span>
              Upload more relevant documents to expand your knowledge base
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground">2.</span>
              Monitor the BM25 index size to ensure documents are properly indexed
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground">3.</span>
              Check Qdrant points count to verify embeddings are stored correctly
            </li>
            <li className="flex items-start gap-2">
              <span className="text-foreground">4.</span>
              Keep an eye on active sessions to monitor system load
            </li>
          </ul>
        </div>
      </div>
    </div>
  )
}

function HealthCard({ label, value }: { label: string; value?: string }) {
  const isHealthy = value === 'healthy' || value === 'ok'
  return (
    <div className="p-4 rounded-lg bg-muted/50">
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <div className="flex items-center gap-2">
        <div
          className={clsx(
            'w-2 h-2 rounded-full',
            isHealthy ? 'bg-green-500' : value ? 'bg-blue-500' : 'bg-muted-foreground'
          )}
        />
        <span className="text-sm font-medium capitalize">
          {value || 'Unknown'}
        </span>
      </div>
    </div>
  )
}

function MetricCard({
  label,
  value,
  icon: Icon,
}: {
  label: string
  value: number | string
  icon: React.ComponentType<{ className?: string }>
}) {
  return (
    <div className="p-4 rounded-lg bg-muted/50">
      <div className="flex items-center gap-2 mb-2">
        <Icon className="w-4 h-4 text-muted-foreground" />
        <span className="text-xs text-muted-foreground">{label}</span>
      </div>
      <p className="text-2xl font-semibold">
        {value}
      </p>
    </div>
  )
}
