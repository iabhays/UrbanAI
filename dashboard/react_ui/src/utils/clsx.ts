import { type ClassValue, clsx as clsxLib } from 'clsx'
import { twMerge } from 'tailwind-merge'

export function clsx(...inputs: ClassValue[]) {
  return twMerge(clsxLib(inputs))
}
