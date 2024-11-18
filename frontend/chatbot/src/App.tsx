import "./App.css";
import { Route, Routes } from "react-router-dom";
import AdminPage from "../src/pages/admin/index";
import Chat from "@pages/chat";
import BoardListPage from "@pages/board";
import LoginForm from "@pages/login";
import { PrivateRoute } from "@util/PrivateRoute";
import TreeView from "@pages/board/TreeView";
import BoardContentTree from "@pages/board/[id]tree";
import RegisterPage from "@pages/signup";
import RegulationPage from "@pages/board/[id]tree";
import TreeCreate from "@pages/board/BoardCreatePage";

import { Provider } from 'react-redux';
import { PersistGate } from 'redux-persist/integration/react';
import { store, persistor } from '@app/store';
import NotificationBound from "@contexts/NotificationBound";

function App() {
	return (
		<>
			<Provider store={store}>
				<PersistGate loading={null} persistor={persistor}>
					<Routes>
						{/* 아래와 같이 prop으로 배경색 동적처리 가능 */}
						<Route path="/" element={<LoginForm />} />
						<Route path="/login" element={<LoginForm />} />
						<Route path="/register" element={<RegisterPage />} />
						<Route element={
							<NotificationBound>
          			<PrivateRoute />
        			</NotificationBound>}>
							<Route path="/boardcontent" element={<BoardContentTree />} />
							<Route path="/adminPage" element={<AdminPage />} />
							<Route path="/chat" element={<Chat />} />
							<Route path="/board" element={<BoardListPage />} />
							<Route
								path="/board/:id"
								element={<RegulationPage/>}
							/>
							{/* <Route path="/tree/:id" element={<TreeView />} /> */}
							<Route path="/treecreate" element={<TreeCreate />} />
						</Route>
						{/* <Route path="/adminPage" element={<AdminPage />} />
						<Route path="/chat" element={<Chat />} />
						<Route path="/board" element={<BoardListPage />} /> */}
					</Routes>
				</PersistGate>
			</Provider>
		</>
		
	);
}

export default App;
