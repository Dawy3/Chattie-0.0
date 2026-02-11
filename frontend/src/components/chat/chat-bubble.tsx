'use client'

import { useState } from 'react'
import { MessageCircle, X } from 'lucide-react'
import { ChatView } from './chat-view'

export function ChatBubble() {
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      {/* Chat Panel */}
      {isOpen && (
        <div className="fixed bottom-[12%] right-6 w-[420px] h-[720px] bg-background border border-border rounded-2xl shadow-2xl flex flex-col overflow-hidden z-50 animate-bubble-open">
          {/* Header */}
          <div className="flex items-center justify-between px-4 py-3 border-b border-border bg-muted/30">
            <div className="flex items-center gap-2">
              <img src="/chat-icon.png" alt="Chat" className="w-8 h-8 rounded-full object-contain bg-white" />
              <div>
                <h3 className="text-sm font-semibold">Chattie</h3>
                <p className="text-xs text-muted-foreground">Your AI chat assistant</p>
              </div>
            </div>
            <button
              onClick={() => setIsOpen(false)}
              className="p-1.5 rounded-lg hover:bg-muted transition-colors"
            >
              <X className="w-4 h-4" />
            </button>
          </div>

          {/* Chat Content */}
          <div className="flex-1 overflow-hidden">
            <ChatView embedded />
          </div>
        </div>
      )}

      {/* Floating Bubble Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="fixed bottom-6 right-6 w-14 h-14 rounded-full bg-white shadow-lg hover:scale-105 active:scale-95 transition-transform flex items-center justify-center z-50"
      >
        {isOpen ? (
          <X className="w-6 h-6 text-foreground" />
        ) : (
          <img src="/chat-icon.png" alt="Chat" className="w-9 h-9 object-contain" />
        )}
      </button>
    </>
  )
}
