export const isAuthenticated = (): boolean => {
	return localStorage.getItem("Authorization") != null;
};
