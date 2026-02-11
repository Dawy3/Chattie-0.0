'use client'

import { ChatBubble } from '@/components/chat/chat-bubble'

export default function Home() {
  return (
    <main className="h-screen flex flex-col items-center justify-center bg-[#2D2D2D]">
      <div className="text-center max-w-2xl px-6">
        <h1 className="text-4xl font-bold mb-4">Knowledge Assistant</h1>
        <p className="text-muted-foreground text-lg mb-8">
          Your AI-powered assistant for querying your document knowledge base.
          Click the chat bubble to get started.
        </p>
      </div>

      <ChatBubble />
    </main>
  )
}
