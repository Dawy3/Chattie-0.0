'use client'

import { useState } from 'react'
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query'
import { FileText, Trash2, RefreshCw, Loader2, FileUp } from 'lucide-react'
import { getDocuments, deleteDocument, type Document } from '@/lib/api'
import { DocumentUpload } from './document-upload'

export function DocumentsView() {
  const [showUpload, setShowUpload] = useState(false)
  const queryClient = useQueryClient()

  const { data, isLoading, refetch } = useQuery({
    queryKey: ['documents'],
    queryFn: () => getDocuments(),
  })

  const deleteMutation = useMutation({
    mutationFn: deleteDocument,
    onMutate: async (documentId) => {
      await queryClient.cancelQueries({ queryKey: ['documents'] })
      const previousData = queryClient.getQueryData(['documents'])

      queryClient.setQueryData(['documents'], (old: typeof data) => {
        if (!old) return old
        return {
          ...old,
          documents: old.documents.filter((doc: Document) => doc.id !== documentId),
          total: old.total - 1,
        }
      })

      return { previousData }
    },
    onError: (error, _documentId, context) => {
      if (context?.previousData) {
        queryClient.setQueryData(['documents'], context.previousData)
      }
      console.error('Delete failed:', error)
      alert(`Failed to delete document: ${error instanceof Error ? error.message : 'Unknown error'}`)
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ['documents'] })
    },
  })

  const handleDelete = async (doc: Document) => {
    if (confirm(`Are you sure you want to delete "${doc.filename}"?`)) {
      deleteMutation.mutate(doc.id)
    }
  }

  return (
    <div className="h-full flex flex-col">
      {/* Header */}
      <div className="border-b border-border p-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-xl font-semibold">Documents</h1>
            <p className="text-sm text-muted-foreground">
              Manage your knowledge base documents
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => refetch()}
              className="p-2 rounded-lg hover:bg-muted transition-colors"
              title="Refresh"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={() => setShowUpload(true)}
              className="flex items-center gap-2 px-4 py-2 bg-foreground text-background rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
            >
              <FileUp className="w-4 h-4" />
              Upload
            </button>
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="flex-1 overflow-y-auto p-4">
        {isLoading ? (
          <div className="flex items-center justify-center h-64">
            <Loader2 className="w-8 h-8 animate-spin text-muted-foreground" />
          </div>
        ) : !data?.documents?.length ? (
          <div className="flex flex-col items-center justify-center h-64 text-center">
            <div className="w-16 h-16 rounded-full bg-muted flex items-center justify-center mb-4">
              <FileText className="w-8 h-8 text-muted-foreground" />
            </div>
            <h2 className="text-lg font-semibold mb-2">No documents yet</h2>
            <p className="text-muted-foreground text-sm mb-4">
              Upload documents to build your knowledge base
            </p>
            <button
              onClick={() => setShowUpload(true)}
              className="flex items-center gap-2 px-4 py-2 bg-foreground text-background rounded-lg text-sm font-medium hover:opacity-90 transition-opacity"
            >
              <FileUp className="w-4 h-4" />
              Upload Document
            </button>
          </div>
        ) : (
          <div className="space-y-2">
            {data.documents.map((doc) => (
              <div
                key={doc.id}
                className="flex items-center gap-4 p-4 rounded-lg border border-border hover:bg-muted/50 transition-colors"
              >
                {/* File icon */}
                <div className="w-10 h-10 rounded-lg bg-muted flex items-center justify-center flex-shrink-0">
                  <FileText className="w-5 h-5 text-muted-foreground" />
                </div>

                {/* Info */}
                <div className="flex-1 min-w-0">
                  <h3 className="font-medium truncate">{doc.filename}</h3>
                  <div className="flex items-center gap-3 text-xs text-muted-foreground mt-1">
                    <span>{doc.chunks_count} chunks</span>
                  </div>
                </div>

                {/* Actions */}
                <button
                  onClick={() => handleDelete(doc)}
                  disabled={deleteMutation.isPending}
                  className="p-2 rounded-lg hover:bg-red-100 text-muted-foreground hover:text-red-600 transition-colors disabled:opacity-50 cursor-pointer"
                  title="Delete document"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Upload Modal */}
      {showUpload && (
        <DocumentUpload
          onClose={() => setShowUpload(false)}
          onSuccess={() => {
            setShowUpload(false)
            queryClient.invalidateQueries({ queryKey: ['documents'] })
          }}
        />
      )}
    </div>
  )
}
