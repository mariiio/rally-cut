import { useState } from 'react';

export function useMenuAnchor() {
  const [anchor, setAnchor] = useState<HTMLElement | null>(null);
  return {
    anchor,
    isOpen: Boolean(anchor),
    open: (e: React.MouseEvent<HTMLElement>) => setAnchor(e.currentTarget),
    close: () => setAnchor(null),
  };
}
