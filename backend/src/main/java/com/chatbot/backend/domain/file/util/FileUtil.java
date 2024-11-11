package com.chatbot.backend.domain.file.util;

import java.io.File;
import java.io.IOException;
import java.util.UUID;

import org.springframework.web.multipart.MultipartFile;

import com.chatbot.backend.domain.file.exception.FileDirectoryInvalidException;

import lombok.extern.slf4j.Slf4j;

@Slf4j
public class FileUtil {
	private static final String DELIMITER = "/";

	/**
	 * 지정된 경로에 디렉토리를 생성
	 * @param basePath 기본 경로
	 * @param dirName 생성할 디렉토리 이름
	 * @return 생성된 디렉토리의 전체 경로
	 */
	public static String createDirectory(String basePath, String dirName) {
		String uploadDir = basePath + DELIMITER + dirName;
		File directory = new File(uploadDir);
		if (!directory.exists()) {
			directory.mkdirs();
		}
		return uploadDir;
	}

	/**
	 * UUID를 사용하여 고유한 파일명을 생성
	 * @param originalFileName 원본 파일명
	 * @return UUID + 원본 확장자
	 */
	public static String generateFileName(String originalFileName) {
		String extension = extractExtension(originalFileName);
		return UUID.randomUUID() + extension;
	}

	/**
	 * 파일명에서 확장자를 추출
	 * @param filename 원본 파일명
	 * @return 확장자
	 */
	public static String extractExtension(String filename) {
		return filename.substring(filename.lastIndexOf("."));
	}

	/**
	 * 전체 파일 경로를 생성
	 * @param uploadDir
	 * @param fileName
	 * @return 전체 파일 경로
	 */
	public static String createFullPath(String uploadDir, String fileName) {
		return uploadDir + DELIMITER + fileName;
	}

	/**
	 * MultipartFile을 실제 파일로 저장
	 * @param file
	 * @param fullPath
	 */
	public static void saveFile(MultipartFile file, String fullPath) throws IOException {
		File dest = new File(fullPath);
		file.transferTo(dest);
	}

	/**
	 * 파일 URL에서 디렉토리 타입을 추출
	 * @param fileUrl
	 * @return
	 */
	public static String extractDirectoryType(String fileUrl) {
		if (fileUrl.contains("/profiles/")) {
			return "profiles";
		} else if (fileUrl.contains("/boards/")) {
			return "boards";
		}
		throw new FileDirectoryInvalidException();
	}

	/**
	 * 파일 URL에서 파일명을 추출
	 * @param fileUrl
	 * @return
	 */
	public static String extractFileName(String fileUrl) {
		String[] fileTokens = fileUrl.split(DELIMITER);
		return fileTokens[fileTokens.length - 1];
	}

	/**
	 * 다운로드를 위한 Content-Disposition 헤더를 생성
	 * @param fileUrl
	 * @return
	 */
	public static String createFileHeader(String fileUrl) {
		return "attachment; filename=\"" + extractFileName(fileUrl) + "\"";
	}

	/**
	 * 디렉토리명과 파일명으로 파일 경로를 생성
	 * @param dirName 디렉토리명
	 * @param fileName 파일명
	 * @return 전체 파일 Path
	 */
	public static String createFilePath(String dirName, String fileName) {
		return dirName + DELIMITER + fileName;
	}
}
