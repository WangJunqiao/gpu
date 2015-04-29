package dd.crawler.codeforce.db;

import java.sql.Connection;
import java.sql.SQLException;
import java.util.ArrayList;
import java.util.List;

import dd.crawler.codeforce.meta.Status;
import dd.crawler.codeforce.meta.Submission;

public class StatusSql {
	private static final String dbName = "10.14.39.53/codeforce";
	private static final String tableName = "status";

	//private static final String dbName = "10.14.39.243/pictune_s";
	//private static final String tableName = "t_flickr_photo";

	private static Connection connection = null;
	//	private static Statement statement;
	static {
		try {
			Class.forName("com.mysql.jdbc.Driver");
			connection = java.sql.DriverManager.getConnection("jdbc:mysql://" + dbName
					+ "?useUnicode=true&characterEncoding=utf-8", "root", "lovelygirl");
			//statement = getStatement();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}

	/**
	 * 测试conn连接是否可用,mysql默认8小时没有活动就关闭连接！！
	 */
	public static boolean testConnection(Connection conn) {
		try {
			conn.setAutoCommit(true);
		} catch (SQLException e) {
			return false;
		}
		return true;
	}

	private static java.sql.Statement getStatement() {
		try {
			return connection.createStatement();
		} catch (SQLException e) {
			e.printStackTrace();
		}
		return null;
	}

	public static void insert(Status status) throws SQLException {
		String sqlString = "insert into " + tableName + " VALUES (" + status.getSubmissionId()
				+ ", '" + status.getContestId() + "', '" + status.getTimestamp() + "', '"
				+ status.getCoderId() + "', '" + status.getLanguage() + "', '"
				+ status.getResult() + "', '" + status.getRunTime() + "', '"
				+ status.getRunMemory() + "')";
		
		getStatement().execute(sqlString);
	}

	public static List<Submission> getStatus(int from, int number) throws SQLException {
		List<Submission> list = new ArrayList<Submission>();
		String sqlString = "SELECT submissionId, contestId from status limit " + from
				+ ", " + number;
		java.sql.ResultSet rs = getStatement().executeQuery(sqlString);
		while (rs.next()) {
			Submission sub = new Submission();
			sub.setSubmissionId(rs.getInt(1));
			sub.setContestId(rs.getInt(2));
			list.add(sub);
		}
		return list;
	}
}
