package edu.zju.dd.codeforce.db;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Connection;
import java.sql.SQLException;

import edu.zju.dd.codeforce.meta.Submission;

public class VisitedSubmissionSql {
	private static final String CODE_PATH = "e:/codeforce-code/";
	private static final String dbName = "localhost/codeforce";
	private static final String tableName = "visited_submissionId";

	private static Connection connection = null;
	static {
		try {
			Class.forName("com.mysql.jdbc.Driver");
			connection = java.sql.DriverManager.getConnection("jdbc:mysql://" + dbName
					+ "?useUnicode=true&characterEncoding=utf-8", "root", "lovelygirl");
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	private static java.sql.Statement getStatement() {
		try {
			return connection.createStatement();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return null;
	}
	
	public static boolean isVisited(int submissionId) throws SQLException {
		java.sql.ResultSet rs = getStatement().executeQuery("select submissionId from "
				+tableName+" where submissionId = "+submissionId);
		return rs.next();
	}
	
	public static void insert(Submission submission) throws SQLException, IOException{
		FileWriter fw = new FileWriter(new File(CODE_PATH+submission.getSubmissionId()+".txt"));
		fw.append("//http://codeforces.com/contest/"+submission.getContestId()+"/submission/"+submission.getSubmissionId()+"\r\n");
		fw.append(submission.getCode());
		fw.close();
		getStatement().execute("insert into " + tableName + " values (" 
				+ submission.getSubmissionId() + ")");
	}
}
