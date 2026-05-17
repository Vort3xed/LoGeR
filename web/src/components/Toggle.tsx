type Props = {
  label: string;
  on: boolean;
  onChange: (v: boolean) => void;
  swatch?: string;
};

export default function Toggle({ label, on, onChange, swatch }: Props) {
  return (
    <label className="flex items-center gap-2 text-sm cursor-pointer">
      <input
        type="checkbox"
        checked={on}
        onChange={(e) => onChange(e.target.checked)}
        className="accent-emerald-500 w-4 h-4"
      />
      {swatch && (
        <span
          className="inline-block w-3 h-3 rounded-sm border border-white/20"
          style={{ background: swatch }}
        />
      )}
      <span>{label}</span>
    </label>
  );
}
