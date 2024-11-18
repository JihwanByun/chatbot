import { defineConfig } from "vite";
import react from "@vitejs/plugin-react-swc";
import path from "path";

// https://vitejs.dev/config/
export default defineConfig({
	plugins: [react()],
	resolve: {
		alias: {
      '@contexts': path.resolve(__dirname, './src/contexts'),
			"@assets": path.resolve(__dirname, "src/assets"),
			"@util": path.resolve(__dirname, "src/util"),
			"@configs": path.resolve(__dirname, "src/configs"),
			"@apis": path.resolve(__dirname, "src/apis"),
			"@components": path.resolve(__dirname, "src/components"),
			"@pages": path.resolve(__dirname, "src/pages/"),
			"@app": path.resolve(__dirname, "src/app"), // 'src' 폴더를 @app 별칭으로 설정
			"@": path.resolve(__dirname, "src"), // '@'를 'src'로 매핑
		},
	},
});
