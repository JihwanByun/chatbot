import React, { useCallback, useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import SembotLayout from "../SembotLayout";
import ButtonOnlyIcon from "@components/atoms/button/ButtonOnlyIcon";
import ButtonWithIcon from "@components/atoms/button/ButtonWithIcon";
import TreeView from "@pages/board/TreeView";
import { useAppDispatch, useAppSelector } from "@app/hooks";
import {
  setRevisionMode,
  startEditNode,
  saveNodeEdit,
} from "@app/slices/treeSlice";
import {
  getBoardDetailAPI,
  downloadFileAPI,
  BoardDetailResponse,
} from "@apis/board/boardDetailApi";
import {
  createFavoriteAPI,
  deleteFavoriteAPI,
} from "@apis/board/boardFavoriteApi";
import ReactMarkdown from "react-markdown";
import { setTreeData } from "@app/slices/treeSlice";
import { ButtonWithIconProps } from "@pages/chat";
import {
  fetchFavoriteBoards,
  updateFavoriteStatus,
} from "@app/slices/favoriteBoardsSlice";
import { logoutUser } from "@app/slices/userSlice";
import { UserRole } from "@util/userConfig";
import BoardUpdate from "@pages/board/BoardUpdate";
import { deleteAlert, errorAlert } from "@util/alert";
import { deleteBoardAPI } from "@apis/board/boardApi";

interface BoardParams {
  id: string;
  [key: string]: string | undefined;
}

const BoardDetailPage: React.FC = () => {
  const { id } = useParams<BoardParams>();
  const navigate = useNavigate();
  const dispatch = useAppDispatch();
  const [boardDetail, setBoardDetail] = useState<BoardDetailResponse>();
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const {role} = useAppSelector(state=>state.users);

  // Redux state
  const isRevisionMode = useAppSelector((state) => state.tree.isRevisionMode);
  const editNodeData = useAppSelector((state) => state.tree.editNodeData);
  const footStyle =
    "flex bg-transparent text-white py-2 px-4 rounded mx-1 transform hover:translate-x-1 transition-all duration-200 cursor-pointer";
  const boardButtonStyle =
    "flex bg-transparent border border-white text-white py-2 px-4 rounded mx-1 hover:bg-blue-900 transition-colors duration-100 ease-in-out";

  const { favorites, loading, hasMore, currentPage } = useAppSelector(
    (state) => state.favoriteBoards
  );

  useEffect(() => {
    if (currentPage === 0) {
      dispatch(fetchFavoriteBoards(0));
    }
    dispatch(setRevisionMode(false))
  }, [dispatch, currentPage]);

  // 더 많은 즐겨찾기 로드
  const handleLoadMore = () => {
    if (!loading && hasMore) {
      dispatch(fetchFavoriteBoards(currentPage + 1));
    }
  };

  // 게시글 상세 정보 조회
  const fetchBoardDetail = useCallback(async () => {
    if (!id) {
      setError("Invalid board ID");
      setIsLoading(false);
      return;
    }

    try {
      setIsLoading(true);
      setError(null);
      const response = await getBoardDetailAPI(Number(id));
      setBoardDetail(response);

      // 규정 데이터를 Redux store에 설정
      dispatch(setTreeData(response.regulationResponseDto));
    } catch (error) {
      console.error("Failed to fetch board detail:", error);
      setError("게시글을 불러오는데 실패했습니다.");
      navigate("/board");
    } finally {
      setIsLoading(false);
    }
  }, [id, dispatch]);

  // 파일 다운로드 처리
  const handleDownloadFile = async () => {
    if (!boardDetail?.fileUrl) return;

    try {
      const blob = await downloadFileAPI(boardDetail.fileUrl);
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      const fileName = boardDetail.fileUrl.split("/").pop() || "download";
      link.download = fileName;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error("Failed to download file:", error);
    }
  };

  // 즐겨찾기 토글
  const handleFavoriteToggle = async () => {
    if (!boardDetail) return;

    try {
      const success = boardDetail.isFavorite
        ? await deleteFavoriteAPI(boardDetail.boardId)
        : await createFavoriteAPI(boardDetail.boardId);

      if (success) {
        // 즐겨찾기 상태 업데이트
        dispatch(
          updateFavoriteStatus({
            boardId: boardDetail.boardId,
            isFavorite: !boardDetail.isFavorite,
            boardData: !boardDetail.isFavorite ? boardDetail : null,
          })
        );

        // 상세 정보 새로고침
        await fetchBoardDetail();
      }
    } catch (error) {
      console.error("Failed to toggle favorite:", error);
    }
  };

  // TreeView 관련 함수들
  const toggleRevisionMode = () => dispatch(setRevisionMode(!isRevisionMode));

  const handleSaveEdit = () => {
    if (editNodeData) {
      dispatch(
        saveNodeEdit()
      );
    }
  };
  
  const handleDelete = () => {    
    try{
      if(boardDetail) deleteBoardAPI(boardDetail?.boardId);
      else throw new Error("게시글 정보가 없습니다.");
    }catch(error){
			if (error instanceof Error) {
				errorAlert(error);
			}
    }finally{
      navigate("/board", { state: { refresh: true } });
    }
  }

  useEffect(() => {
    fetchBoardDetail();
  }, [fetchBoardDetail,isRevisionMode]);

  const getChildren = () => (
    isRevisionMode ? boardDetail &&
    <>
      <BoardUpdate board={boardDetail} onUpdate={toggleRevisionMode} onBackClick={()=>dispatch(setRevisionMode(false))} />
    </>
    : 
    <div className="bg-white rounded-lg px-6 py-2 space-y-6 text-left">
      {/* 게시글 헤더 */}
      <div className="space-y-4">
        <div className="flex items-center gap-4 relative">
          <h1 className="text-xl font-semibold">{boardDetail?.title}</h1>
          <span className="px-2 py-1 rounded-full text-xs font-medium bg-gray-100 whitespace-nowrap">
            답변 레벨 : {boardDetail?.level}
          </span>
          {/* 상단 버튼 영역 */}
          <div className="absolute flex items-end space-x-1 right-0">
            {role === UserRole.USER_WRITE && (
              <>
                <ButtonOnlyIcon
                  key="edit"
                  icon="/src/assets/icons/pen.svg"
                  styleName="p-2 hover:bg-gray-100 rounded"
                  width={18}
                  onClick={toggleRevisionMode}
                />
                <ButtonOnlyIcon
                  key="delete"
                  icon="/src/assets/icons/delete-black.svg"
                  styleName="p-2 hover:bg-gray-100 rounded"
                  width={18}
                  onClick={()=>deleteAlert(handleDelete)}
                />
              </>
            )}
            <ButtonOnlyIcon
              key="favorite"
              icon={
                boardDetail?.isFavorite
                  ? "/src/assets/icons/favorited.svg"
                  : "/src/assets/icons/favorite.svg"
              }
              width={18}
              styleName="p-2 hover:bg-gray-100 rounded"
              onClick={handleFavoriteToggle}
            />
          </div>
        </div>

        <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
          <div className="flex items-center space-x-2">
            <img
              className="h-8 w-8 rounded-full"
              src={boardDetail?.writer.profileUrl}
              alt={boardDetail?.writer.name}
            />
            <span className="font-medium">{boardDetail?.writer.name}</span>
            <span className="text-gray-500 text-xs">
              &lt;{boardDetail?.writer.email}&gt;
            </span>
          </div>
          <div className="flex flex-col items-start sm:items-end">
            <span className="text-xs text-gray-500">
              {boardDetail && new Date(boardDetail.createdAt).toLocaleString()}
            </span>
            {boardDetail?.hasFile && (
              <ButtonWithIcon
                btnName="파일 다운로드"
                styleName="mt-2 text-sm text-blue-600 hover:text-blue-800"
                icon="/src/assets/icons/document-download.svg"
                handleClick={handleDownloadFile}
              />
            )}
          </div>
        </div>
      </div>

      <hr className="border-gray-200" />

      {/* 게시글 내용 - Markdown 렌더링 */}
      <div
        className="prose max-w-none text-left
                [&>*]:text-left 
                prose-headings:text-left
                prose-p:text-left
                prose-ul:text-left
                prose-ol:text-left"
      >
        {/* 게시글 내용 - Markdown 렌더링 */}
        {boardDetail?.contents && (
          <ReactMarkdown>
            {boardDetail.contents.replace(/\\n/g, "\n")}
          </ReactMarkdown>
        )}
      </div>

      {/* 규정 트리뷰 */}
      {boardDetail?.regulationResponseDto && (
        <div className="mt-8">
          <TreeView
            isRevisionMode={isRevisionMode}
            handleStartEdit={(node) =>
              dispatch(
                startEditNode({
                  id: node.id,
                  title: node.title || "",
                  content: node.content || "",
                })
              )
            }
            handleSaveEdit={handleSaveEdit}
          />
        </div>
      )}
    </div>
  );

  // 사이드바 컴포넌트 구성
  const sidebarComponents: ButtonWithIconProps[] = [
    {
      btnName: "규정 목록",
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text-footer.svg",
      handleClick: () => navigate("/board"),
    },
    // 즐겨찾기 목록을 버튼 컴포넌트로 변환
    ...favorites.map((favorite) => ({
      btnName: favorite.title,
      styleName: boardButtonStyle,
      icon: "/src/assets/icons/book-open-text-footer.svg",
      handleClick: () => navigate(`/board/${favorite.boardId}`),
    })),
  ];

  const footerComponents = [
    {
      btnName: "채팅",
      styleName: footStyle,
      icon: "/src/assets/icons/chatting-icon.svg",
      handleClick: () => {
        navigate("/chat");
      },
    },
    {
      btnName: "로그아웃",
      styleName: footStyle,
      icon: "/src/assets/icons/logout.svg",
      handleClick: async () => {
        await dispatch(logoutUser());
        navigate("/");
      },
    },
  ];

  if (error) {
    return (
      <SembotLayout
        title="오류"
        sidebarComponents={sidebarComponents}
        footerComponents={footerComponents}
      >
        <div className="flex justify-center items-center h-full">
          <div className="text-red-500">{error}</div>
        </div>
      </SembotLayout>
    );
  }

  if (isLoading) {
    return (
      <SembotLayout
        title="로딩 중..."
        sidebarComponents={sidebarComponents}
        footerComponents={footerComponents}
      >
        <div className="flex justify-center items-center h-full">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-gray-900"></div>
        </div>
      </SembotLayout>
    );
  }

  return (
    <SembotLayout
      title={"규정 상세"}
      sidebarComponents={sidebarComponents}
      footerComponents={footerComponents}
      onLoadMore={handleLoadMore}
      children={getChildren()}
    />
  );
};

export default BoardDetailPage;
