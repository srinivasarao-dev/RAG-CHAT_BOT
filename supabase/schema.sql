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

-- ---------------------------------------------------------------- chat with files (RAG)
create extension if not exists vector;

create table if not exists public.files (
    id               uuid primary key default gen_random_uuid(),
    user_id          text not null,
    conversation_id  uuid references public.conversations(id) on delete cascade,
    name             text not null,
    mime             text,
    size             integer,
    content          text,            -- full extracted text
    n_chunks         integer,
    created_at       timestamptz not null default now()
);

create table if not exists public.file_chunks (
    id           bigint generated always as identity primary key,
    file_id      uuid not null references public.files(id) on delete cascade,
    chunk_index  integer not null,
    content      text not null,
    embedding    vector(384) not null   -- BAAI/bge-small-en-v1.5, computed locally by the app
);

create index if not exists idx_files_conversation on public.files (conversation_id);
create index if not exists idx_files_user on public.files (user_id, created_at desc);
create index if not exists idx_file_chunks_file on public.file_chunks (file_id);
create index if not exists idx_file_chunks_embedding on public.file_chunks using hnsw (embedding vector_cosine_ops);

-- Semantic search over the chunks of the given files
create or replace function public.match_file_chunks(query_embedding vector(384), file_ids uuid[], match_count int default 6)
returns table (file_id uuid, file_name text, chunk_index int, content text, score float)
language sql stable
as $$
    select c.file_id, f.name, c.chunk_index, c.content, 1 - (c.embedding <=> query_embedding) as score
    from public.file_chunks c
    join public.files f on f.id = c.file_id
    where c.file_id = any(file_ids)
    order by c.embedding <=> query_embedding
    limit match_count;
$$;

-- Row Level Security: enabled with no public policies, so the anon key can't read anyone's chats.
-- The app connects with the service_role key (server side only, never expose it in a browser),
-- which bypasses RLS. If you later add Supabase Auth, add per-user policies on user_id = auth.uid().
alter table public.conversations enable row level security;
alter table public.messages      enable row level security;
alter table public.tool_calls    enable row level security;
alter table public.files         enable row level security;
alter table public.file_chunks   enable row level security;
