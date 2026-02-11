'use client'

import { useState, useRef, useEffect, KeyboardEvent } from 'react'
import { Send, Loader2 } from 'lucide-react'
import { useChatStore, useSettingsStore } from '@/lib/store'
import { streamQuery } from '@/lib/api'

interface ChatInputProps {
  conversationId: string
  suggestion?: string
  onSuggestionUsed?: () => void
}

export function ChatInput({ conversationId, suggestion, onSuggestionUsed }: ChatInputProps) {
  const [input, setInput] = useState('')
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  const { addMessage, updateMessage, appendToMessage, isStreaming, setStreaming, getCurrentConversation } = useChatStore()
  const { temperature, maxTokens, topK } = useSettingsStore()

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto'
      textareaRef.current.style.height = `${Math.min(textareaRef.current.scrollHeight, 200)}px`
    }
  }, [input])

  useEffect(() => {
    if (suggestion) {
      onSuggestionUsed?.()
      handleSubmit(suggestion)
    }
  }, [suggestion])

  const handleSubmit = async (directMessage?: string) => {
    const messageToSend = directMessage || input.trim()
    if (!messageToSend || isStreaming) return

    const userMessage = messageToSend
    setInput('')
    setStreaming(true)

    // Add user message
    addMessage(conversationId, {
      role: 'user',
      content: userMessage,
    })

    // Add placeholder for assistant message
    const assistantMessageId = addMessage(conversationId, {
      role: 'assistant',
      content: '',
    })

    try {
      const conversation = getCurrentConversation()
      const sessionId = conversation?.sessionId || conversationId

      const stream = streamQuery({
        query: userMessage,
        session_id: sessionId,
        top_k: topK,
        temperature,
        max_tokens: maxTokens,
        use_history: true,
      })

      for await (const event of stream) {
        switch (event.type) {
          case 'metadata':
            updateMessage(conversationId, assistantMessageId, {
              sources: event.data.sources,
              cached: event.data.cache,
              route: event.data.route,
              sessionId: event.data.session_id,
            })
            break

          case 'token':
            appendToMessage(conversationId, assistantMessageId, event.data.content)
            break

          case 'done':
            updateMessage(conversationId, assistantMessageId, {
              model: event.data.model,
              latencyMs: event.data.latency_ms,
            })
            break

          case 'error':
            updateMessage(conversationId, assistantMessageId, {
              content: `Error: ${event.data.detail}`,
            })
            break
        }
      }
    } catch (error) {
      console.error('Failed to send message:', error)
      updateMessage(conversationId, assistantMessageId, {
        content: 'Sorry, there was an error processing your request. Please try again.',
      })
    } finally {
      setStreaming(false)
    }
  }

  const handleKeyDown = (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  return (
    <div className="border-t border-border p-4">
      <div className="flex items-end gap-3">
        <div className="flex-1 relative">
          <textarea
            ref={textareaRef}
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="اضف سؤالك هنا"
            disabled={isStreaming}
            rows={1}
            className="w-full resize-none rounded-xl border border-border bg-background px-4 py-3 pr-12 text-sm focus:outline-none focus:ring-2 focus:ring-foreground/20 disabled:opacity-50"
          />
          <button
            onClick={() => handleSubmit()}
            disabled={!input.trim() || isStreaming}
            className="absolute right-2 bottom-2 p-2 rounded-lg bg-foreground text-background disabled:opacity-50 disabled:cursor-not-allowed hover:opacity-90 transition-opacity"
          >
            {isStreaming ? (
              <Loader2 className="w-4 h-4 animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </div>
      </div>
      <p className="text-xs text-muted-foreground text-center mt-2">
        Chattie can make mistakes
      </p>
    </div>
  )
}
