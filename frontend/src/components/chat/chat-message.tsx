'use client'

import { useState } from 'react'
import { User, Bot, ChevronDown, ChevronUp, Zap } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import { clsx } from 'clsx'
import type { ConversationMessage } from '@/lib/store'

interface ChatMessageProps {
  message: ConversationMessage
}

export function ChatMessage({ message }: ChatMessageProps) {
  const [showSources, setShowSources] = useState(false)
  const isUser = message.role === 'user'

  return (
    <div
      className={clsx(
        'flex gap-3',
        isUser ? 'flex-row-reverse' : 'flex-row'
      )}
    >
      {/* Avatar */}
      <div
        className={clsx(
          'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
          isUser ? 'bg-foreground' : 'bg-muted'
        )}
      >
        {isUser ? (
          <User className="w-4 h-4 text-background" />
        ) : (
          <Bot className="w-4 h-4 text-foreground" />
        )}
      </div>

      {/* Content */}
      <div
        className={clsx(
          'flex flex-col max-w-[75%]',
          isUser ? 'items-end' : 'items-start'
        )}
      >
        <div
          className={clsx(
            'px-4 py-3 rounded-2xl',
            isUser
              ? 'bg-foreground text-background rounded-tr-md'
              : 'bg-muted rounded-tl-md'
          )}
        >
          {isUser ? (
            <p className="text-sm whitespace-pre-wrap">{message.content}</p>
          ) : message.content ? (
            <div className="prose prose-sm max-w-none dark:prose-invert">
              <ReactMarkdown>{message.content}</ReactMarkdown>
            </div>
          ) : (
            <div className="flex items-center gap-1.5 py-1 px-1">
              <span className="w-2 h-2 bg-foreground/70 rounded-full loading-dot" />
              <span className="w-2 h-2 bg-foreground/70 rounded-full loading-dot" />
              <span className="w-2 h-2 bg-foreground/70 rounded-full loading-dot" />
            </div>
          )}
        </div>

        {/* Metadata for assistant messages */}
        {!isUser && (
          <div className="flex items-center gap-3 mt-2 text-xs text-muted-foreground">
            {message.model && (
              <span className="flex items-center gap-1">
                <Zap className="w-3 h-3" />
                {message.model}
              </span>
            )}
          </div>
        )}

        {/* Sources */}
        {!isUser && message.sources && message.sources.length > 0 && (
          <div className="mt-2 w-full">
            <button
              onClick={() => setShowSources(!showSources)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              {showSources ? (
                <ChevronUp className="w-3 h-3" />
              ) : (
                <ChevronDown className="w-3 h-3" />
              )}
              {message.sources.length} source{message.sources.length !== 1 ? 's' : ''}
            </button>
            {showSources && (
              <div className="mt-2 space-y-2">
                {message.sources.map((source, index) => (
                  <div
                    key={source.chunk_id || index}
                    className="p-3 bg-muted rounded-lg text-xs"
                  >
                    <div className="flex items-center justify-between mb-1">
                      <span className="font-medium">
                        Source {index + 1}
                      </span>
                      <span className="text-muted-foreground">
                        Score: {(source.score * 100).toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-muted-foreground line-clamp-3">
                      {source.content}
                    </p>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  )
}
