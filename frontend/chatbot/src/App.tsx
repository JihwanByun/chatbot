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
import BoardEdit from "@pages/board/BoardEdit";
import TreeView from "@pages/board/TreeView";
import BoardContentTree from "@pages/board/[id]tree";
import SignUp from "@pages/signup/index";

function App() {
	const [curPage, setCurPage] = useState(4);
	return (
		<>
			<Routes>
				{/* 아래와 같이 prop으로 배경색 동적처리 가능 */}
				<Route path="/" element={<LoginForm />} />
				<Route path="/login" element={<LoginForm />} />
				<Route element={<PrivateRoute />}>
					<Route path="/boardcontent" element={<BoardContentTree />} />
					<Route path="/adminPage" element={<AdminPage />} />
					<Route path="/chat" element={<Chat />} />
					<Route path="/board" element={<BoardListPage />} />
					<Route path="/tree" element={<TreeView />} />
				</Route>
				{/* <Route path="/adminPage" element={<AdminPage />} />
				<Route path="/chat" element={<Chat />} />
				<Route path="/board" element={<BoardListPage />} /> */}
				<Route path="/" element={<Devpage />} />
				<Route
					path="/button"
					element={
						<ButtonPrimary
							btnName="Hello"
							styleName="bg-slate-500 rounded-lg py-3 px-6 font-sans text-xs font-bold uppercase text-white shadow-md hover:shadow-lg"
						/>
					}
				/>
				<Route
					path="/buttonicon"
					element={
						<ButtonWithIcon
							btnName="Hello"
							styleName="flex items-center justify-center middle none center rounded-lg bg-slate-300 py-3 px-6 font-sans text-xs font-bold uppercase text-white shadow-md shadow-pink-500/20 transition-all hover:shadow-lg hover:shadow-pink-500/40 focus:opacity-[0.85] focus:shadow-none active:opacity-[0.85] active:shadow-none disabled:pointer-events-none disabled:opacity-50 disabled:shadow-none"
							icon={plus}
						/>
					}
				/>
				<Route
					path="/paging"
					element={
						<Paging curPage={curPage} totalPage={8} onPageChange={setCurPage} />
					}
				/>

				<Route path="/adminPage" element={<AdminPage />} />
				<Route path="/chat" element={<Chat />} />
				<Route path="/board" element={<BoardListPage />} />
				<Route path="/login" element={<LoginForm />} />
				<Route path="/register" element={<SignUp />} />
			</Routes>
		</>
	);
}

export default App;
