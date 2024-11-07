import "./App.css";
import { Route, Routes } from "react-router-dom";
import ButtonPrimary from "./components/atoms/button/ButtonPrimary";
import ButtonWithIcon from "./components/atoms/button/ButtonWithIcon";
import plus from "./assets/icons/plus.svg";
import Devpage from "./Devpage";
import Paging from "@components/atoms/paging/Paging";
import AdminPage from "../src/pages/admin/index";
import { useState } from "react";
import Chat from "@pages/chat";
import BoardListPage from "@pages/board";
import LoginForm from "@pages/login";
import { PrivateRoute } from "@util/PrivateRoute";

function App() {
	const [curPage, setCurPage] = useState(4);
	return (
		<>
			<Routes>
				{/* 아래와 같이 prop으로 배경색 동적처리 가능 */}
				<Route path="/" element={<LoginForm />} />
				<Route path="/login" element={<LoginForm />} />
				<Route element={<PrivateRoute />}>
					<Route path="/adminPage" element={<AdminPage />} />
					<Route path="/chat" element={<Chat />} />
					<Route path="/board" element={<BoardListPage />} />
				</Route>
				{/* <Route path="/adminPage" element={<AdminPage />} />
				<Route path="/chat" element={<Chat />} />
				<Route path="/board" element={<BoardListPage />} /> */}
			</Routes>
		</>
	);
}

export default App;
