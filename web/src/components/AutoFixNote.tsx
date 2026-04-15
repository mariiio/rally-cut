import type { AutoFix } from '@/types/rally';

export function AutoFixNote({ fixes }: { fixes: AutoFix[] | undefined }) {
  if (!fixes || fixes.length === 0) return null;
  return (
    <ul
      style={{
        margin: '4px 0 0 0',
        padding: 0,
        listStyle: 'none',
        fontSize: 12,
        color: '#4caf50',
      }}
    >
      {fixes.map((fx) => (
        <li key={fx.id}>✓ {fx.message}</li>
      ))}
    </ul>
  );
}
