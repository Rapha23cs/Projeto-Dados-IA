import { createClient } from '@supabase/supabase-js'

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

// Inicializa o cliente do Supabase apenas se as chaves estiverem presentes
export const supabase = (supabaseUrl && supabaseAnonKey && supabaseUrl !== 'COLE_SUA_URL_DO_SUPABASE_AQUI') 
  ? createClient(supabaseUrl, supabaseAnonKey) 
  : null;
