import React from "react";

interface AvatarProps {
  name: string;
  size?: string;
  className?: string;
}

const getRandomPastelColor = (str: string) => {
  let hash = 0;
  for (let i = 0; i < str.length; i++) {
    hash = str.charCodeAt(i) + ((hash << 5) - hash);
  }
  const h = hash % 360;
  return `hsl(${h}, 70%, 85%)`;
};

export const Avatar: React.FC<AvatarProps> = ({ name, size = "2.5rem", className = "" }) => {
  const initial = name.charAt(0).toUpperCase();
  const backgroundColor = getRandomPastelColor(name);

  return (
    <div
      className={`rounded-full flex items-center justify-center ${className}`}
      style={{
        backgroundColor,
        color: "#444",
        width: size,
        height: size,
        fontSize: `calc(${size} * 0.4)`,
        fontWeight: "600",
      }}
    >
      {initial}
    </div>
  );
};
