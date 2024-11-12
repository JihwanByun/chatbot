import "./App.css";
import { Route, Routes } from "react-router-dom";
import AdminPage from "../src/pages/admin/index";
import { useState } from "react";
import Chat from "@pages/chat";
import BoardListPage from "@pages/board";
import LoginForm from "@pages/login";
import { PrivateRoute } from "@util/PrivateRoute";
import TreeView from "@pages/board/TreeView";
import BoardContentTree from "@pages/board/[id]tree";
import RegisterPage from "@pages/signup";

function App() {
	return (
		<>
			<Routes>
				{/* 아래와 같이 prop으로 배경색 동적처리 가능 */}
				<Route path="/" element={<LoginForm />} />
				<Route path="/login" element={<LoginForm />} />
				<Route path="/register" element={<RegisterPage />} />
				<Route element={<PrivateRoute />}>
					<Route path="/boardcontent" element={<BoardContentTree />} />
					<Route path="/adminPage" element={<AdminPage />} />
					<Route path="/chat" element={<Chat />} />
					<Route path="/board" element={<BoardListPage />} />
					<Route path="/board/:id" element={<RegulationPage title="규정 정보" />} />
					<Route path="/tree" element={<TreeView />} />
				</Route>
				{/* <Route path="/adminPage" element={<AdminPage />} />
				<Route path="/chat" element={<Chat />} />
				<Route path="/board" element={<BoardListPage />} /> */}
			</Routes>
		</>
	);
}

export default App;
