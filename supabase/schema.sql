-- Run this once in Supabase: Dashboard -> SQL Editor -> New query -> paste -> Run.

create extension if not exists "pgcrypto";

create table if not exists public.conversations (
    id          uuid primary key default gen_random_uuid(),
    user_id     text not null,
    title       text not null default 'New chat',
    created_at  timestamptz not null default now(),
    updated_at  timestamptz not null default now()
);

create table if not exists public.messages (
    id               bigint generated always as identity primary key,
    conversation_id  uuid not null references public.conversations(id) on delete cascade,
    role             text not null check (role in ('user', 'assistant')),
    content          text not null,
    metadata         jsonb not null default '{}'::jsonb,   -- tools used, sources, model, latency
    created_at       timestamptz not null default now()
);

-- Every external API call the bot makes: useful for debugging, analytics and cost tracking
create table if not exists public.tool_calls (
    id               bigint generated always as identity primary key,
    conversation_id  uuid references public.conversations(id) on delete cascade,
    tool             text not null,
    args             jsonb,
    output_preview   text,
    latency_ms       integer,
    created_at       timestamptz not null default now()
);

create index if not exists idx_conversations_user on public.conversations (user_id, updated_at desc);
create index if not exists idx_messages_conversation on public.messages (conversation_id, id);
create index if not exists idx_tool_calls_tool on public.tool_calls (tool, created_at desc);

-- Row Level Security: enabled with no public policies, so the anon key can't read anyone's chats.
-- The app connects with the service_role key (server side only, never expose it in a browser),
-- which bypasses RLS. If you later add Supabase Auth, add per-user policies on user_id = auth.uid().
alter table public.conversations enable row level security;
alter table public.messages      enable row level security;
alter table public.tool_calls    enable row level security;
