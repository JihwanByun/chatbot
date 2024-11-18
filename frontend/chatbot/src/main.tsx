// import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { BrowserRouter } from "react-router-dom";
import { Provider } from "react-redux";
import App from "./App.tsx";
import "./index.css";
import store from "./app/store.ts";

import { ToastContainer } from 'react-toastify';
import 'react-toastify/dist/ReactToastify.css';
import { NotificationProvider } from "@contexts/NotificationContext.tsx";

createRoot(document.getElementById("root")!).render(
	// <StrictMode>
	<BrowserRouter>
		<NotificationProvider>
			<Provider store={store}>
				<App />
				<ToastContainer />
			</Provider>
		</NotificationProvider>
	</BrowserRouter>
	// </StrictMode>
);
