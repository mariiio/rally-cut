export default function AppLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <div style={{ height: '100%', overflow: 'hidden' }}>
      {children}
    </div>
  );
}
